from .esn import *
from .scr import *
from .gpflowmodel import *
from .robustgpmodel import *
import numpy as np
import GPy
import GPyOpt
import copy
import json
import pyDOE
from tqdm import tqdm
from collections import OrderedDict


__all__ = ['EchoStateNetworkCV']


class EchoStateNetworkCV:
    """A cross-validation object that automatically optimizes ESN hyperparameters using Bayesian optimization with
    Gaussian Process priors. 
    
    Searches optimal solution within the provided bounds.
    
    Parameters
    ----------
    bounds : dict
        A dictionary specifying the bounds for optimization. The key is the parameter name and the value 
        is a tuple with minimum value and maximum value of that parameter. E.g. {'n_nodes': (100, 200), ...}
    model : class: {EchoStateNetwork, SimpleCycleReservoir}
            Model class to optimize
    subsequence_length : int
        Number of samples in one cross-validation sample
    eps : float
        The number specifying the maximum amount of change in parameters before considering convergence
    initial_samples : int (minimum 30)
        The number of random samples to explore the  before starting optimization
    validate_fraction : float
        The fraction of the data that may be used as a validation set
    steps_ahead : int or None
        Number of steps to use in n-step ahead prediction for cross validation. `None` indicates prediction 
        of all values in the validation array.
    max_iterations : int
        Maximim number of iterations in optimization
    batch_size : int
        Batch size of samples used by GPyOpt
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
    esn_feedback : bool or None
        Build ESNs with feedback ('teacher forcing') or not
    verbose : bool
        Verbosity on or off
    
    """
    
    def __init__(self, bounds, subsequence_length, model=EchoStateNetwork, eps=1e-8, initial_samples=8, 
                 validate_fraction=0.2, steps_ahead=1, max_iterations=1000, batch_size=1, cv_samples=1, 
                 mcmc_samples=None, scoring_method='nmse', tanh_alpha=1., esn_burn_in=100, acquisition_type='LCB',
                 max_time=np.inf, n_jobs=1, random_seed=42, esn_feedback=None, verbose=True):
        # Bookkeeping
        self.bounds = OrderedDict(bounds)  # Fix order
        self.parameters = list(self.bounds.keys())
        self.indices = dict(zip(self.parameters, range(len(self.parameters))))  # Parameter indices
        
        # Store settings
        self.model = model
        self.subsequence_length = subsequence_length
        self.eps = eps
        self.initial_samples = initial_samples
        self.validate_fraction = validate_fraction
        self.steps_ahead = steps_ahead
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
        self.feedback = esn_feedback
        self.verbose = verbose
        
        # Normalize bounds domains and remember transformation
        self.scaled_bounds, self.bound_scalings, self.bound_intercepts = self.normalize_bounds(self.bounds)
        
        # Build constraints based on bounds
        self.constraints = None  # self.build_constraints(self.bounds)
            
    def build_constraints(self, bounds):
        """Builds GPy style constraints for the optimization"""
        constraints = []
        
        # Set contraint (spectral radius - leaking rate ≤ 0)
        if 'leaking_rate' in bounds and 'spectral_radius' in bounds:
            spectral_index = self.indices['spectral_radius']
            leaking_index = self.indices['leaking_rate']
            
            # Adjust for domain scaling
            spectral_scale = self.bound_scalings[spectral_index]
            leaking_scale = self.bound_scalings[leaking_index]
            
            # Adjust for intercepts
            spectral_intercept = self.bound_intercepts[spectral_index]
            leaking_intercept = self.bound_intercepts[leaking_index]
            
            # Add constraint in GPy format
            constraints += [{'name': 'Spectral ≤ leaking', 'constrain': '{} * x[:, {}] + {} - ({} * x[:, {}] + {})'.format(
                spectral_scale, spectral_index, spectral_intercept, leaking_scale, leaking_index, leaking_intercept
            )}]
        
        # Set contraint (expected connections in reservoir larger than 1)
        if 'n_nodes' in bounds and 'connectivity' in bounds:
            nodes_index = self.indices['n_nodes']
            connect_index = self.indices['connectivity']
            
            # Adjust for domain scaling
            nodes_scale = self.bound_scalings[nodes_index]
            connect_scale = self.bound_scalings[connect_index]
            
            # Adjust for intercepts
            nodes_intercept = self.bound_intercepts[nodes_index]
            connect_intercept = self.bound_intercepts[connect_index]
            
            # Add constraint in GPy format
            constraints += [{'name': '-Expected connections ≤ -1', 'constrain': '-({} * x[:, {}] + {}) * ({} * x[:, {}] + {})'.format(
                nodes_scale, nodes_index, nodes_intercept, connect_scale, connect_index, connect_intercept
            )}]
            
        return constraints if len(constraints) > 0 else None
                                   
    
    def normalize_bounds(self, bounds):
        """Makes sure all bounds feeded into GPyOpt are scaled to the domain [0, 1], 
        to aid interpretation of convergence plots. 
        
        Scalings are saved in instance parameters.
        
        Parameters
        ----------
        bounds : dicts
            Contains dicts with boundary information
            
        Returns
        -------
        scaled_bounds, scalings, intercepts : tuple
            Contains scaled bounds (list of dicts in GPy style), the scaling applied (numpy array) 
            and an intercept (numpy array) to transform values back to their original domain
            
        """
        scaled_bounds = []
        scalings = []
        intercepts = []
        for name, domain in self.bounds.items():
            # Get scaling
            lower_bound = min(domain)
            upper_bound = max(domain)
            scale = upper_bound - lower_bound
            
            # Transform to [0, 1] domain
            scaled_bound = {'name': name, 'type': 'continuous', 'domain': (0., 1.)}
            
            # Store
            scaled_bounds.append(scaled_bound)
            scalings.append(scale)
            intercepts.append(lower_bound)
        return scaled_bounds, np.array(scalings), np.array(intercepts)
    
    def denormalize_bounds(self, normalized_arguments):
        """Denormalize arguments to feed into model.
        
        Parameters
        ----------
        normalized_arguments : numpy array
            Contains arguments in same order as bounds
            
        Returns
        -------
        denormalized_arguments : 1-D numpy array
            Array with denormalized arguments
        
        """
        denormalized_bounds = (normalized_arguments.ravel() * self.bound_scalings) + self.bound_intercepts
        return denormalized_bounds
        
    def construct_arguments(self, x):
        """Constructs arguments for ESN input from input array.
        
        Does so by denormalizing and adding arguments not involved in optimization,
        like the random seed.
        
        Parameters
        ----------
        x : 1-D numpy array
            Array containing normalized parameter values
        
        Returns
        -------
        arguments : dict
            Arguments that can be fed into an ESN
            
        """
        # Denormalize        
        denormalized_values = self.denormalize_bounds(x)        
        arguments = dict(zip(self.parameters, denormalized_values))
        
        # Specific edits
        arguments['random_seed'] = self.seed
        if 'regularization' in arguments:
            arguments['regularization'] = 10. ** arguments['regularization']  # Log scale   
        
        if 'n_nodes' in arguments:
            arguments['n_nodes'] = int(np.round(arguments['n_nodes']))
        
        if not self.feedback is None:
            arguments['feedback'] = self.feedback
            
        return arguments
    
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
            tqdm.write("Warning: y-array has more series (columns) than samples (rows). Check if this is correct")
        
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
        kernel = GPy.kern.Matern52(input_dim=len(self.parameters), ARD=True)
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
            #keyword_arguments['evaluator_type'] = 'local_penalization'  # BUG: convergence is not consistent
            pass
        
        if self.verbose:
            tqdm.write("Model initialization and exploration run...")
        
        # Build optimizer    
        self.optimizer = GPyOpt.methods.BayesianOptimization(f=self.objective_sampler,
                                                             domain=self.scaled_bounds,
                                                             initial_design_numdata=self.initial_samples,
                                                             constrains=self.constraints, 
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
            tqdm.write("Model initialization done.", '\n')
            tqdm.write(self.optimizer.model.model, '\n')
            tqdm.write(self.optimizer.model.model.kern.lengthscale, '\n')
        
        if self.verbose:
            tqdm.write("Starting optimization...")
        
        # Optimize
        self.optimizer.run_optimization(eps=self.eps, max_iter=self.max_iterations, max_time=self.max_time, 
                                        verbosity=self.verbose)
        
        # Inform user
        if self.verbose:        
            tqdm.write('Done.')
            
        # Purge temporary data references
        del self.x
        del self.y
        
        # Store in dict
        best_arguments = self.construct_arguments(self.optimizer.x_opt)
        
        # Save to disk if desired
        if not store_path is None:
            with open(store_path, 'w+') as output_file:
                json.dump(best_arguments, output_file, indent=4)
        
        # Show convergence
        self.optimizer.plot_convergence()
        
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
            tqdm.write("Model initialization and exploration run...")
        
        
        # Define objective
        objective = GPyOpt.core.task.SingleObjective(self.objective_sampler, 
                                                     objective_name='ESN Objective',
                                                     batch_type='synchronous',
                                                     num_cores=self.n_jobs)
        
        # Set search space and constraints
        space = GPyOpt.core.task.space.Design_space(self.scaled_bounds, self.constraints)
        
        # Select model and acquisition
        if self.mcmc_samples is None:
            acquisition_type = self.acquisition_type
            model = RobustGPModel(normalize_Y=True)
        else:
            acquisition_type = self.acquisition_type + '_MCMC'
            # Set GP kernel
            kernel = GPy.kern.Matern52(input_dim=len(self.parameters), ARD=True)
            
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
                                                       leapfrog_steps=5, 
                                                       verbose=self.verbose)
            # Set prior on Gaussian Noise # BUG
            #model.model.likelihood.variance.set_prior(gamma_prior())
        
        # Explicitly state model
        if self.verbose:
            tqdm.write('Using model:', model.__class__.__name__, '\n')
        
        # Set acquisition
        acquisition_optimizer = GPyOpt.optimization.AcquisitionOptimizer(space, optimizer='lbfgs')
        SelectedAcquisition = GPyOpt.acquisitions.select_acquisition(acquisition_type)
        acquisition = SelectedAcquisition(model=model, space=space, optimizer=acquisition_optimizer)
        
        # Add Local Penalization
        lp_acquisition = GPyOpt.acquisitions.LP.AcquisitionLP(model, space, acquisition_optimizer, acquisition, transform='none')
        
        # Set jitter to low number if used
        try:
            acquisition.jitter = 1e-6
        except AttributeError:
            pass
        
        # Set initial design
        n = len(self.parameters)
        # noise_estiation_parameters = np.random.uniform(-1e-6, 1e-6, size=(30, n)) + 0.5
        # random_samples = np.random.uniform(size=(self.initial_samples, n))
        # initial_parameters = np.vstack((noise_estiation_parameters, random_samples))
        initial_parameters = pyDOE.lhs(n, self.initial_samples, 'cm') # Latin hypercube initialization
        
        # Pick evaluator
        evaluator = GPyOpt.core.evaluators.batch_local_penalization.LocalPenalization(acquisition=lp_acquisition, 
                                                                                      batch_size=self.batch_size, 
                                                                                      normalize_Y=True)
        # Show progress bar
        if self.verbose:
            tqdm.write("Starting optimization...")
            self.pbar = tqdm(total=self.max_iterations, unit=' objective evalutions')
        
        # Build optimizer
        update_interval = 1 if self.mcmc_samples is None else 20
        self.optimizer = GPyOpt.methods.ModularBayesianOptimization(model=model, space=space, objective=objective, 
                                                                    acquisition=lp_acquisition, evaluator=evaluator,  # N.B. LP acquisition!
                                                                    X_init=initial_parameters, normalize_Y=True, 
                                                                    model_update_interval=update_interval)
        self.optimizer.modular_optimization = True
                                     
        # Show model
        if self.verbose:
            tqdm.write("Model initialization done.", '\n')
            tqdm.write(self.optimizer.model.model, '\n')
            tqdm.write(self.optimizer.model.model.kern.lengthscale, '\n')
        
        # Optimize
        self.optimizer.run_optimization(eps=self.eps, 
                                        max_iter=self.max_iterations, 
                                        max_time=self.max_time, 
                                        verbosity=self.verbose)
        
        # Inform user
        if self.verbose:        
            tqdm.write('Done.')
            self.pbar.close()
            
        # Purge temporary data references
        del self.x
        del self.y
        
        # Show convergence
        if not store_path is None:
            plot_path = store_path[-5] + '_convergence.png'
        else:
            plot_path = None
        self.optimizer.plot_convergence(filename=plot_path)
            
        # Store in dict
        best_arguments = self.construct_arguments(self.optimizer.x_opt)
        
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
        arguments = self.construct_arguments(parameters)

        # Build network
        esn = self.model(**arguments)
        # Train
        esn.train(x=train_x, y=train_y, burn_in=self.esn_burn_in)

        # Validation score
        score = esn.test(x=validate_x, y=validate_y, scoring_method=self.scoring_method, 
                         steps_ahead=self.steps_ahead, alpha=self.alpha)
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
            start_index = np.random.randint(viable_start, viable_stop)
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
            self.pbar.update(1)
            pars = self.construct_arguments(parameters)
            self.pbar.set_postfix(**{'\nCurrent score:': mean_score, '\n': ''}, **pars)
            
        # Return scores
        return mean_score.reshape(-1, 1)
