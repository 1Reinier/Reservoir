from .esn import *
from .gpflowmodel import *
import numpy as np
import GPy
import GPyOpt
import copy
import json


__all__ = ['EchoStateNetworkCV']


class EchoStateNetworkCV:
    """A cross-validation object that automatically optimizes ESN hyperparameters using Bayesian optimization with
    Gaussian Process priors. 
    
    Searches optimal solution within the provided bounds.
    
    Parameters
    ----------
    bounds : list
        List of dicts specifying the bounds for optimization. Every dict has to contain entries for the following keys:
        - name : string
        - type : {'continuous', 'discrete'}
        - domain : tuple
              Tuple with minimum value and maximum value of that parameter.
    subsequence_length : int
        Number of samples in one cross-validation sample
    eps : float
        The number specifying the maximum amount of change in parameters before considering convergence
    initial_samples : int
        The number of random samples to explore the  before starting optimization
    validate_fraction : float
        The fraction of the data that may be used as a validation set
    max_iterations : int
        Maximim number of iterations in optimization
    batch_size : int
        Batch size of samples used by GPyOpt (currently not working due to a bug in GPyOpt)
    cv_samples : int
        Number of samples of the objective function to evaluate for a given parametrization of the ESN
    mcmc_samples : int or None
        If any number of samples is specified, GPyOpt will use MCMC to draw samples for optimization
        of the Gaussian Process. This may make ESN optimization more accurate but can also slow it down.
    scoring_method : {'mse', 'rmse', 'tanh', 'nmse', 'nrmse', 'log', 'log-tanh', 'tanh-nrmse'}
        Evaluation metric that is used to guide optimization
    tanh_alpha : float
        Alpha coefficient used to scale the tanh error function: alpha * tanh{(1 / alpha) * mse}
    esn_burn_in : int
        Number of time steps to dicard upon training a single Echo State Network
    acquisition_type : {'MPI', 'EI', 'LCB'}
        The type of acquisition function to use in Bayesian Optimization
    max_time : float
        Maximum number of seconds before quitting optimization
    n_jobs : int
        Maximum number of concurrent jobs
    verbose : bool
        Verbosity on or off
    
    """
    
    def __init__(self, bounds, subsequence_length, eps=1e-8, initial_samples=8, validate_fraction=0.2, 
                 max_iterations=1000, batch_size=1, cv_samples=1, mcmc_samples=None, scoring_method='tanh-nrmse', 
                 tanh_alpha=3., esn_burn_in=100, acquisition_type='LCB', max_time=np.inf, n_jobs=1, 
                 random_seed=42, verbose=True):
        self.bounds = bounds
        self.subsequence_length = subsequence_length
        self.eps = eps
        self.initial_samples = initial_samples
        self.validate_fraction = validate_fraction
        self.max_iterations = max_iterations
        self.batch_size = batch_size
        self.cv_samples = cv_samples
        self.mcmc_samples = mcmc_samples
        self.scoring_method = scoring_method
        self.alpha = tanh_alpha
        self.esn_burn_in = esn_burn_in
        self.acquisition_type = acquisition_type
        self.max_time = max_time
        self.n_jobs = n_jobs
        self.seed = random_seed
        self.verbose = verbose
        
        # Normalize bounds domains and remember transformation
        self.scaled_bounds, self.bound_scalings, self.bound_intercepts = self.normalize_bounds(self.bounds)
            
    def normalize_bounds(self, bounds):
        """Makes sure all bounds feeded into GPyOpt are scaled to the domain [0, 1], 
        to aid interpretation of convergence plots. 
        
        Scalings are saved in instance parameters.
        
        Parameters
        ----------
        bounds : list of dicts
            Contains dicts of GPyOpt boundary information
            
        Returns
        -------
        scaled_bounds, bound_scalings, bound_intercepts : tuple
            Contains scaled bounds (dict), the scaling applied (numpy array) and an intercept 
            (numpy array) to transform values back to their original domain
            
        """
        scaled_bounds = []
        bound_scalings = []
        bound_intercepts = []
        for bound in bounds:
            # Get scaling
            lower_bound = min(bound['domain'])
            upper_bound = max(bound['domain'])
            scale = upper_bound - lower_bound
            
            # Transform
            scaled_bound = copy.deepcopy(bound)
            scaled_bound['domain'] = tuple(map(lambda value: (value - lower_bound) / scale, bound['domain']))
            
            # Store
            scaled_bounds.append(scaled_bound)
            bound_scalings.append(scale)
            bound_intercepts.append(lower_bound)
        return scaled_bounds, np.array(bound_scalings), np.array(bound_intercepts)
    
    def denormalize_bounds(self, normalized_arguments):
        """Denormalize arguments to feed into model.
        
        Parameters
        ----------
        normalized_arguments : 1-D numpy array
            Contains arguments in same order as bounds
            
        Returns
        -------
        denormalized_arguments : 1-D numpy array
            Array with denormalized arguments
        
        """
        denormalized_bounds = (normalized_arguments * self.bound_scalings) + self.bound_intercepts
        return denormalized_bounds
        
    def validate_data(self, y, x=None, verbose=True):
        """Validates inputted data against errors in shape and common mistakes.
        
        Parameters
        ----------
        y : numpy array
            A y-array to be checked (should be 2-D with series in columns)
        x : numpy array or None
            Optional x-array to be checked (should have same number of rows as y)
        verbose: bool
            Toggle to flag printed messages about common shape issues
            
        Raises
        ------
        ValueError
            Throws ValueError when data is not in the correct format.
            
        """
        # Check dimensions
        if not y.ndim == 2:
            raise ValueError("y-array is not 2 dimensional")
        
        if verbose and y.shape[0] < y.shape[1]:
            print("Warning: y-array has more series (columns) than samples (rows). Check if this is correct")
        
        # Checks for x
        if not x is None:
            
            # Check dimensions
            if not x.ndim == 2:
                raise ValueError("x-array is not 2 dimensional")
            
            # Check shape equality
            if x.shape[0] != y.shape[0]:
                raise ValueError("y-array and x-array have different number of samples (rows)")    
    
    def optimize(self, y, x=None, store_path=None):
        """Performs optimization (with cross-validation).
        
        Uses Bayesian Optimization with Gaussian Process priors to optimize ESN hyperparameters.
        
        Parameters
        ----------
        y : numpy array
            Array with target values (y-values)
        
        x : numpy array or None
            Optional array with input values (x-values)
        
        store_path : str or None
            Optional path where to store best found parameters to disk (in JSON)
        
        Returns
        -------
        best_arguments : numpy array
            The best parameters found during optimization
        
        """
        # Checks
        self.validate_data(y, x, self.verbose)
        
        # Initialize new random state
        self.random_state = np.random.RandomState(self.seed + 2)
        
        # Temporarily store the data
        self.x = x
        self.y = y
        
        # Keywords to feed into Bayesian Optimization
        gamma_prior = lambda: GPy.priors.Gamma(1., 1.)  
        kernel = GPy.kern.Matern52(input_dim=7, ARD=True)
        kernel.variance.set_prior(gamma_prior())
        kernel.lengthscale.set_prior(gamma_prior())

        keyword_arguments = {'kernel': kernel}
        
        if self.mcmc_samples is None:
            # MLE solution
            model_type = 'GP'
            completed_acquisition_type = self.acquisition_type
        else:
            # MCMC solution
            model_type = 'GP_MCMC'
            completed_acquisition_type = self.acquisition_type + '_MCMC'
            keyword_arguments['n_samples'] = self.mcmc_samples
            
        if self.batch_size > 1:
            # Add more exploration if batch size is larger than 1
            #keyword_arguments['evaluator_type'] = 'local_penalization'  # BUG: GPyOpt cannot parallelize this
            pass
            
        # Set contraint (spectral radius - leaking rate ≤ 0)
        constraints = [{'name': 'alpha-rho', 'constrain': 'x[:, 3] - x[:, 2]'}]
        
        if self.verbose:
            print("Model initialization and exploration run...")
        
        # Build optimizer    
        self.optimizer = GPyOpt.methods.BayesianOptimization(f=self.objective_sampler,
                                                             domain=self.scaled_bounds,
                                                             initial_design_numdata=self.initial_samples,
                                                             constrains=constraints, 
                                                             model_type=model_type, 
                                                             acquisition_type=completed_acquisition_type,
                                                             exact_feval=False,
                                                             normalize_Y=True,
                                                             cost_withGradients=None,
                                                             acquisition_optimizer_type='lbfgs',
                                                             verbosity=self.verbose,
                                                             num_cores=self.n_jobs, 
                                                             batch_size=self.batch_size,
                                                             **keyword_arguments) 
                                                             
                                                             
        # Show model
        if self.verbose:
            print("Model initialization done.", '\n')
            print(self.optimizer.model.model, '\n')
            print(self.optimizer.model.model.kern.lengthscale, '\n')
        
        if self.verbose:
            print("Starting optimization...")
        
        # Optimize
        self.optimizer.run_optimization(eps=self.eps, max_iter=self.max_iterations, max_time=self.max_time, 
                                        verbosity=self.verbose)
        
        # Inform user
        if self.verbose:        
            print('Done.')
            
        # Purge temporary data references
        del self.x
        del self.y
        
        # Show convergence
        self.optimizer.plot_convergence()
        
        # Scale arguments
        best_found = self.denormalize_bounds(self.optimizer.x_opt).T
        
        # Store in dict
        best_arguments = dict(input_scaling=best_found[0], feedback_scaling=best_found[1], leaking_rate=best_found[2], 
                         spectral_radius=best_found[3], regularization=10.**best_found[4], connectivity=10.**best_found[5], 
                         n_nodes=best_found[6], random_seed=self.seed, feedback=True)
        
        # Save to disk if desired
        if not store_path is None:
            with open(store_path, 'w+') as output_file:
                json.dump(best_arguments, output_file, indent=4)
        
        # Return best parameters
        return best_arguments
        
    def optimize_modular(self, y, x=None, store_path=None):
        """Performs optimization (with cross-validation).
        
        Uses Bayesian Optimization with Gaussian Process priors to optimize ESN hyperparameters.
        
        Parameters
        ----------
        y : numpy array
            Array with target values (y-values)
        
        x : numpy array or None
            Optional array with input values (x-values)
        
        store_path : str or None
            Optional path where to store best found parameters to disk (in JSON)
        
        Returns
        -------
        best_arguments : numpy array
            The best parameters found during optimization
        
        """
        # Checks
        self.validate_data(y, x, self.verbose)
        
        # Initialize new random state
        self.random_state = np.random.RandomState(self.seed + 2)
        
        # Temporarily store the data
        self.x = x
        self.y = y
        
        # Inform user    
        if self.verbose:
            print("Model initialization and exploration run...")
        
        
        # Define objective
        objective = GPyOpt.core.task.SingleObjective(self.objective_sampler, 
                                                     objective_name='ESN Objective',
                                                     batch_type='synchronous',
                                                     num_cores=self.n_jobs)
        
        # Set search space and constraints (spectral radius - leaking rate ≤ 0)
        constraints = [{'name': 'alpha-rho', 'constrain': 'x[:, 3] - x[:, 2]'}]
        space = GPyOpt.core.task.space.Design_space(self.bounds, constraints)
        
        # Select model and acquisition
        if self.mcmc_samples is None:
            acquisition_type = self.acquisition_type
            model = GPflowModel(normalize_Y=True)
        else:
            acquisition_type = self.acquisition_type + '_MCMC'
            # Set GP kernel
            kernel = GPy.kern.Matern52(input_dim=7, ARD=True)
            
            # Proper distribution close to Jeffrey's prior
            gamma_prior = lambda: GPy.priors.Gamma(.001, .001)  
            kernel.variance.set_prior(gamma_prior())
            kernel.lengthscale.set_prior(gamma_prior())
            
            # MCMC Model
            model = GPyOpt.models.gpmodel.GPModel_MCMC(kernel=kernel,  
                                                       noise_var=None, 
                                                       exact_feval=False, 
                                                       normalize_Y=True, 
                                                       n_samples=self.mcmc_samples, 
                                                       n_burnin=100, 
                                                       subsample_interval=2, 
                                                       step_size=0.2, 
                                                       leapfrog_steps=20, 
                                                       verbose=self.verbose)
            # Set prior on Gaussian Noise
            #model.model.likelihood.variance.set_prior(gamma_prior())
        
        # Set acquisition TODO: Local Penalization
        acquisition_optimizer = GPyOpt.optimization.AcquisitionOptimizer(space, optimizer='lbfgs')
        SelectedAcquisition = GPyOpt.acquisitions.select_acquisition(acquisition_type)
        acquisition = SelectedAcquisition(model=model, space=space, optimizer=acquisition_optimizer)
        try:
            # Set jitter to 0 if used
            acquisition.jitter = 1e-6
        except AttributeError:
            pass
        
        # Set initial design
        initial_x = GPyOpt.util.stats.sample_initial_design('latin', space, self.initial_samples)  # Latin hypercube initialization
        
        # Pick evaluator
        evaluator = GPyOpt.core.evaluators.Predictive(acquisition=acquisition, batch_size=self.batch_size, normalize_Y=True)
        
        # Build optimizer
        update_interval = 1 if self.mcmc_samples is None else 20
        self.optimizer = GPyOpt.methods.ModularBayesianOptimization(model=model, space=space, objective=objective, 
                                                                    acquisition=acquisition, evaluator=evaluator, 
                                                                    X_init=initial_x, normalize_Y=True, 
                                                                    model_update_interval=update_interval)
                                     
        # Show model
        if self.verbose:
            print("Model initialization done.", '\n')
            print(self.optimizer.model.model, '\n')
        
        if self.verbose:
            print("Starting optimization...")
        
        # Optimize
        self.optimizer.run_optimization(eps=self.eps, max_iter=self.max_iterations, max_time=self.max_time, 
                                        verbosity=self.verbose)
        
        # Inform user
        if self.verbose:        
            print('Done.')
            
        # Purge temporary data references
        del self.x
        del self.y
        
        # Show convergence
        self.optimizer.plot_convergence(filename=store_path + '.convergence.png')
        
        # Scale arguments
        best_found = self.denormalize_bounds(self.optimizer.x_opt).T
        
        # Store in dict
        best_arguments = dict(input_scaling=best_found[0], feedback_scaling=best_found[1], leaking_rate=best_found[2], 
                         spectral_radius=best_found[3], regularization=10.**best_found[4], connectivity=10.**best_found[5], 
                         n_nodes=best_found[6], random_seed=self.seed, feedback=True)
        
        # Save to disk if desired
        if not store_path is None:
            with open(store_path, 'w+') as output_file:
                json.dump(best_arguments, output_file, indent=4)
        
        # Return best parameters
        return best_arguments
        
    def objective_function(self, parameters, train_y, validate_y, train_x=None, validate_x=None):
        """Returns selected error metric on validation set.
        
        Parameters
        ----------
        parameters : array
            Parametrization of the Echo State Network, in column vector shape: (n, 1).
        train_y : array
            Dependent variable of the training set
        validate_y : array
            Dependent variable of the validation set
        train_x : array or None
            Independent variable(s) of the training set
        validate_x : array or None
            Independent variable(s) of the validation set
        
        Returns
        -------
        score : float
            Score on provided validation set
        
        """
        # Get arguments
        arguments = self.denormalize_bounds(parameters).T

        # Build network
        esn = EchoStateNetwork(input_scaling=arguments[0], feedback_scaling=arguments[1], leaking_rate=arguments[2], 
                               spectral_radius=arguments[3], regularization=10.**arguments[4], 
                               connectivity=10.**arguments[5], n_nodes=arguments[6], random_seed=self.seed, 
                               feedback=True)
        # Train
        esn.train(x=train_x, y=train_y, burn_in=self.esn_burn_in)

        # Validation score
        score = esn.test(x=validate_x, y=validate_y, scoring_method=self.scoring_method, alpha=self.alpha)
        return score

    def objective_sampler(self, parameters):
        """Splits training set into train and validate sets, and computes multiple samples of the objective function.
        
        This method also deals with dispatching multiple series to the objective function if there are multiple,
        and aggregates the returned scores by averaging.
        
        Parameters
        ----------
        parameters : array
            Parametrization of the Echo State Network
        
        Returns
        -------
        mean_score : 2-D array
            Column vector with mean score(s), as required by GPyOpt
        
        """
        # Get data
        training_y = self.y
        training_x = self.x
        
        # Get number of series
        n_series = training_y.shape[1]
        
        # Set viable sample range
        viable_start = self.esn_burn_in
        viable_stop = training_y.shape[0] - self.subsequence_length
        
        # Get sample lengths
        validate_length = np.round(self.subsequence_length * self.validate_fraction).astype(int)
        train_length = self.subsequence_length - validate_length
        
        # Score storage
        scores = np.zeros((self.cv_samples, n_series))
        
         # Get samples
        for i in range(self.cv_samples):  # TODO: Can be parallelized
            
            # Get indices
            start_index = self.random_state.randint(viable_start, viable_stop)
            train_stop_index = start_index + train_length
            validate_stop_index = train_stop_index + validate_length
            
            # Get samples
            train_y = training_y[start_index: train_stop_index]
            validate_y = training_y[train_stop_index: validate_stop_index]
            
            if not training_x is None:
                train_x = training_x[start_index: train_stop_index]
                validate_x = training_x[train_stop_index: validate_stop_index]
            else:
                train_x = None
                validate_x = None
            
            # Loop through series and score result
            for n in range(n_series):
                scores[i, n] = self.objective_function(parameters, train_y[:, n].reshape(-1, 1), 
                                                       validate_y[:, n].reshape(-1, 1), train_x, validate_x)
        
        # Pass back as a column vector (as required by GPyOpt)
        mean_score = scores.mean()
        
        # Inform user
        if self.verbose:    
            print('Objective mean:', mean_score, 'Scores:', scores.ravel())
            
        # Return scores
        return mean_score.reshape(-1, 1) 
