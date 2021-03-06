from .esn import *
from .scr import *
from .detail.robustgpmodel import *
from .detail.esn_bo import *
import numpy as np
import GPy
import GPyOpt
import copy
import json
import pyDOE
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
    initial_samples : int
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
    scoring_method : {'mse', 'rmse', 'tanh', 'nmse', 'nrmse', 'log', 'log-tanh', 'tanh-nrmse'}
        Evaluation metric that is used to guide optimization
    log_space : bool
        Optimize in log space or not (take the logarithm of the objective or not before modeling it in the GP)
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
        Build ESNs with feedback ('teacher forcing') if available
    update_interval : int (default 1)
        After how many acquisitions the GPModel should be updated
    verbose : bool
        Verbosity on or off
    plot : bool
        Show convergence plot at end of optimization
    target_score : float
        Quit when reaching this target score

    """

    def __init__(self, bounds, subsequence_length, model=EchoStateNetwork, eps=1e-8, initial_samples=50,
                 validate_fraction=0.2, steps_ahead=1, max_iterations=1000, batch_size=1, cv_samples=1,
                 scoring_method='nmse', log_space=True, tanh_alpha=1., esn_burn_in=100, acquisition_type='LCB',
                 max_time=np.inf, n_jobs=1, random_seed=123, esn_feedback=None, update_interval=1, verbose=True,
                 plot=True, target_score=0.):
        # Bookkeeping
        self.bounds = OrderedDict(bounds)  # Fix order
        self.parameters = list(self.bounds.keys())
        self.free_parameters = []
        self.fixed_parameters = []

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
        self.scoring_method = scoring_method
        self.log_space = log_space
        self.alpha = tanh_alpha
        self.esn_burn_in = np.int32(esn_burn_in)
        self.acquisition_type = acquisition_type
        self.max_time = max_time
        self.n_jobs = n_jobs
        self.seed = random_seed
        self.feedback = esn_feedback
        self.update_interval = update_interval
        self.verbose = verbose
        self.plot = plot
        self.target_score = target_score

        # Normalize bounds domains and remember transformation
        self.scaled_bounds, self.bound_scalings, self.bound_intercepts = self.normalize_bounds(self.bounds)

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
            # Get any fixed parmeters
            if type(domain) == int or type(domain) == float:
                # Take note
                self.fixed_parameters.append(name)

            # Free parameters
            elif type(domain) == tuple:
                # Bookkeeping
                self.free_parameters.append(name)

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
            else:
                raise ValueError("Domain bounds not understood")

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
        # Denormalize free parameters
        denormalized_values = self.denormalize_bounds(x)
        arguments = dict(zip(self.free_parameters, denormalized_values))

        # Add fixed parameters
        for name in self.fixed_parameters:
            value = self.bounds[name]
            arguments[name] = value

        # Specific additions
        arguments['random_seed'] = self.seed
        if 'regularization' in arguments:
            arguments['regularization'] = 10. ** arguments['regularization']  # Log scale correction

        if 'connectivity' in arguments:
            arguments['connectivity'] = 10. ** arguments['connectivity']  # Log scale correction

        if 'n_nodes' in arguments:
            arguments['n_nodes'] = int(np.round(arguments['n_nodes']))  # Discretize

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
            Column vector with target values (y-values)

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
        self.x = x.astype(np.float32) if x is not None else None
        self.y = y.astype(np.float32)

        # Inform user
        if self.verbose:
            print("Model initialization and exploration run...")

        # Define objective
        objective = GPyOpt.core.task.SingleObjective(self.objective_sampler,
                                                     objective_name='ESN Objective',
                                                     batch_type='synchronous',
                                                     num_cores=self.n_jobs)

        # Set search space and constraints
        space = GPyOpt.core.task.space.Design_space(self.scaled_bounds, constraints=None)

        # Select model and acquisition
        acquisition_type = self.acquisition_type
        model = RobustGPModel(normalize_Y=True, log_space=self.log_space)

        # Set acquisition
        acquisition_optimizer = GPyOpt.optimization.AcquisitionOptimizer(space, optimizer='lbfgs')
        SelectedAcquisition = GPyOpt.acquisitions.select_acquisition(acquisition_type)
        acquisition = SelectedAcquisition(model=model, space=space, optimizer=acquisition_optimizer)

        # Add Local Penalization
        # lp_acquisition = GPyOpt.acquisitions.LP.AcquisitionLP(model, space, acquisition_optimizer, acquisition,
        # transform='none')

        # Set initial design
        n = len(self.free_parameters)
        initial_parameters = pyDOE.lhs(n, self.initial_samples, 'm')  # Latin hypercube initialization

        # Pick evaluator
        if self.batch_size == 1:
            evaluator = GPyOpt.core.evaluators.sequential.Sequential(acquisition=acquisition,
                                                                     batch_size=self.batch_size)
        else:
            evaluator = GPyOpt.core.evaluators.RandomBatch(acquisition=acquisition,
                                                           batch_size=self.batch_size)
        # Show progress bar
        if self.verbose:
            print("Starting optimization...", '\n')

        # Build optimizer
        self.optimizer = EchoStateBO(model=model, space=space, objective=objective,
                                     acquisition=acquisition, evaluator=evaluator,
                                     X_init=initial_parameters, model_update_interval=self.update_interval)

        # Optimize
        self.iterations_taken = self.optimizer.run_target_optimization(target_score=self.target_score,
                                                                       eps=self.eps,
                                                                       max_iter=self.max_iterations,
                                                                       max_time=self.max_time,
                                                                       verbosity=self.verbose)

        # Inform user
        if self.verbose:
            print('Done after', self.iterations_taken, 'iterations.')

        # Purge temporary data references
        del self.x
        del self.y

        # Show convergence
        if not store_path is None:
            plot_path = store_path[-5] + '_convergence.png'
        else:
            plot_path = None

        if self.plot or not store_path is None:
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
        scores = np.zeros((self.cv_samples, n_series), dtype=np.float32)

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
            print('Score:', mean_score)
            # pars = self.construct_arguments(parameters)

        # Return scores
        return mean_score.reshape(-1, 1)
