import numpy as np
import scipy.stats
import scipy.linalg
import GPy
import GPyOpt
import copy
import json


__all__ = ['EchoStateNetwork', 'EchoStateNetworkCV']


class EchoStateNetwork:
    """Class with all functionality to train Echo State Nets.
    
    Builds and echo state network with the specified parameters.
    In training, testing and predicting, x is a matrix consisting of column-wise time series features. 
    Y is a zero-dimensional target vector.
    
    Parameters
    ----------
    n_nodes : int
        Number of nodes that together make up the reservoir
    input_scaling : float
        The scaling of input values into the network
    feedback_scaling : float
        The scaling of feedback values back into the reservoir
    spectral_radius : float
        Sets the magnitude of the largest eigenvalue of the transition matrix (weight matrix)
    leaking_rate : float
        Specifies how much of the state update 'leaks' into the new state
    connectivity : float
        The probability that two nodes will be connected
    regularization : float
        The L2-regularization parameter used in Ridge regression for model inference
    feedback : bool
        Sets feedback of the last value back into the network on or off
    random_seed : int
        Seed used to initialize RandomState in reservoir generation and weight initialization
        
    Methods
    -------
    train(y, x=None, burn_in=100)
        Train an Echo State Network
    test(y, x=None, y_start=None, scoring_method='mse', alpha=1.)
        Tests and scores against known output
    predict(n_steps, x=None, y_start=None)
        Predicts n values in advance
    predict_stepwise(y, x=None, steps_ahead=1, y_start=None)
        Predicts a specified number of steps into the future for every time point in y-values array
    
    """
    
    def __init__(self, n_nodes, input_scaling=0.5, feedback_scaling=0.5, spectral_radius=0.8, 
                 leaking_rate=1.0, connectivity=0.1, regularization=1e-8, feedback=True, random_seed=42):
        # Parameters
        self.n_nodes = int(np.round(n_nodes))
        self.input_scaling = input_scaling
        self.feedback_scaling = feedback_scaling
        self.spectral_radius = spectral_radius
        self.connectivity = connectivity
        self.input_scaling = input_scaling
        self.leaking_rate = leaking_rate
        self.regularization = regularization
        self.feedback = feedback
        self.seed = random_seed
        self.generate_reservoir()

        
    def generate_reservoir(self):
        """Generates random reservoir from parameters set at initialization."""
        # Initialize new random state
        random_state = np.random.RandomState(self.seed)
        
        # Set weights and sparsity randomly
        max_tries = 1000
        for i in range(max_tries):
            self.weights = random_state.uniform(-1., 1., size=(self.n_nodes, self.n_nodes))
            accept = random_state.uniform(size=(self.n_nodes, self.n_nodes)) < self.connectivity
            self.weights *= accept
        
            # Set spectral density
            max_eigenvalue = np.abs(np.linalg.eigvals(self.weights)).max()
            if max_eigenvalue > 0:
                break
            elif i == max_tries - 1:
                raise ValueError('Nilpotent reservoirs are not allowed. \
                                 Increase connectivity and/or number of nodes.')
        
        # Set spectral radius of weight matrix
        self.weights *= self.spectral_radius / max_eigenvalue
        
        # Default state
        self.state = np.zeros((1, self.n_nodes))
        
        # Set out to none to indicate untrained ESN
        self.out_weights = None
        
    def draw_reservoir(self):
        """Vizualizes reservoir. 
        
        Requires 'networkx' package.
        
        """
        import networkx as nx
        graph = nx.DiGraph(self.weights)
        nx.draw(graph)
        
    def normalize(self, inputs=None, outputs=None, keep=False):
        """Normalizes array by column (along rows) and stores mean and standard devation.
        
        Set `store` to True if you want to retain means and stds for denormalization later.
        
        Parameters
        ----------
        inputs : array or None
            Input matrix that is to be normalized
        outputs : array or None
            Output column vector that is to be normalized
        keep : bool
            Stores the normalization transformation in the object to denormalize later
        
        Returns
        -------
        transformed : tuple or array
            Returns tuple of every normalized array. In case only one object is to be returned the tuple will be 
            unpacked before returning
        
        """      
        # Checks
        if inputs is None and outputs is None:
            raise ValueError('Inputs and outputs cannot both be None')
        
        # Storage for transformed variables
        transformed = []
        
        if not inputs is None:
            if keep:
                # Store for denormalization
                self._input_means = inputs.mean(axis=0)
                self._input_stds = inputs.std(ddof=1, axis=0)
                
            # Transform
            transformed.append((inputs - self._input_means) / self._input_stds)
            
        if not outputs is None:
            if keep:
                # Store for denormalization
                self._output_means = outputs.mean(axis=0)
                self._output_stds = outputs.std(ddof=1, axis=0)

            # Transform
            transformed.append((outputs - self._output_means) / self._output_stds)
        
        # Syntactic sugar
        return tuple(transformed) if len(transformed) > 1 else transformed[0]
        
    def denormalize(self, inputs=None, outputs=None):
        """Denormalizes array by column (along rows) using stored mean and standard deviation.
        
        Parameters
        ----------
        inputs : array or None
            Any inputs that need to be transformed back to their original scales
        outputs : array or None
            Any output that need to be transformed back to their original scales
        
        Returns
        -------
        transformed : tuple or array
            Returns tuple of every denormalized array. In case only one object is to be returned the tuple will be 
            unpacked before returning
        
        """
        if inputs is None and outputs is None:
            raise ValueError('Inputs and outputs cannot both be None')
        
        # Storage for transformed variables
        transformed = []
        
        if not inputs is None:
            transformed.append((inputs * self._input_stds) + self._input_means)
        if not outputs is None:
            transformed.append((outputs * self._output_stds) + self._output_means)
        
        # Syntactic sugar
        return tuple(transformed) if len(transformed) > 1 else transformed[0]
    
    def train(self, y, x=None, burn_in=100):
        """Trains the Echo State Network.

        Trains the out weights on the random network. This is needed before being able to make predictions.
        Consider running a burn-in of a sizable length. This makes sure the state  matrix has converged to a 
        'credible' value.
        
        Parameters
        ----------
        y : array
            Column vector of y values
        x : array or None
            Optional matrix of inputs (features by column)
        burn_in : int
            Number of inital time steps to be discarded for model inference
        
        Returns
        -------
        complete_data, y, burn_in : tuple
            Returns the complete dataset (state matrix concatenated with any feedback and/or inputs),
            the y values provided and the number of time steps used for burn_in. These data can be used
            for diagnostic purposes  (e.g. vizualization of activations).
        
        """
        # Checks
        if x is None and not self.feedback:
            raise ValueError("Error: provide x or enable feedback")
        
        # Initialize new random state
        random_state = np.random.RandomState(self.seed + 1)
            
        # Normalize inputs and outputs
        y = self.normalize(outputs=y, keep=True)
        if not x is None:
            x = self.normalize(inputs=x, keep=True)
        
        # Reset state
        current_state = self.state[-1]  # From default or pretrained state
        
        # Calculate correct shape based on feedback (feedback means one row less)
        rows = y.shape[0] - 1 if self.feedback else y.shape[0]
        start_index = 1 if self.feedback else 0  # Convenience index
        
        # Build state matrix
        self.state = np.zeros((rows, self.n_nodes))
        
        # Build inputs
        inputs = np.ones((rows, 1))  # Add bias for all t = 0, ..., T
        
        # Add data inputs if present
        if not x is None:
            inputs = np.hstack((inputs, x[start_index:]))  # Add data inputs
            
        # Set and scale input weights (for memory length and non-linearity)
        self.in_weights = self.input_scaling * random_state.uniform(-1, 1, size=(self.n_nodes, inputs.shape[1]))
                
        # Add feedback if requested, optionally with feedback scaling
        if self.feedback:
            inputs = np.hstack((inputs, y[:-1]))  # Add teacher forced signal (equivalent to y(t-1) as input)
            feedback_weights = self.feedback_scaling * random_state.uniform(-1, 1, size=(self.n_nodes, 1))
            self.in_weights = np.hstack((self.in_weights, feedback_weights))
                
        # Train iteratively
        for t in range(inputs.shape[0]):
            update = np.tanh(self.in_weights @ inputs[t].T + self.weights @ current_state)
            current_state = self.leaking_rate * update + (1 - self.leaking_rate) * current_state  # Leaking separate
            self.state[t] = current_state
        
        # Concatenate inputs with node states
        complete_data = np.hstack((inputs, self.state))
        train_x = complete_data[burn_in:]  # Include everything after burn_in
        train_y = y[burn_in + 1:] if self.feedback else y[burn_in:]
        
        # Ridge regression, full inverse solution
        self.out_weights = np.linalg.inv(train_x.T @ train_x + self.regularization * \
                                        np.eye(train_x.shape[1])) @ train_x.T @ train_y 

        # Store last y value as starting value for predictions
        self.y_last = y[-1]
        
        # Return all data for computation or visualization purposes (Note: these are normalized)
        return complete_data, (y[1:] if self.feedback else y), burn_in
            
    def test(self, y, x=None, y_start=None, scoring_method='mse', alpha=1.):
        """Tests and scores against known output.
        
        Parameters
        ----------
        y : array
            Column vector of known outputs
        x : array or None
            Any inputs if required
        y_start : float or None
            Starting value from which to start testing. If None, last stored value from trainging will be used
        scoring_method : {'mse', 'rmse', 'nrmse', 'tanh'}
            Evaluation metric used to calculate error
        alpha : float
            Alpha coefficient to scale the tanh error transformation: alpha * tanh{(1 / alpha) * error}
            
        Returns
        -------
        error : float
            Error between prediction and knwon outputs
        
        """
        # Run prediction
        y_predicted = self.predict(y.shape[0], x, y_start=y_start)
        
        # Return error
        return self.error(y_predicted, y, scoring_method, alpha=alpha)
    
    def predict(self, n_steps, x=None, y_start=None):
        """Predicts n values in advance.
        
        Prediction starts from the last state generated in training.
        
        Parameters
        ----------
        n_steps : int
            The number of steps to predict into the future (internally done in one step increments)
        x : numpy array or None
            If prediciton requires inputs, provide them here
        y_start : float or None
            Starting value from which to start prediction. If None, last stored value from training will be used
        
        Returns
        -------
        y_predicted : numpy array
            Array of n_step predictions
        
        """
        # Check if ESN has been trained
        if self.out_weights is None or self.y_last is None:
            raise ValueError('Error: ESN not trained yet')
        
        # Normalize the inputs (like was done in train)
        if not x is None:
            x = self.normalize(inputs=x)
            
        # Initialize input
        inputs = np.ones((n_steps, 1))  # Add bias term
        
        # Choose correct input
        if x is None and not self.feedback:
            raise ValueError("Error: cannot run without feedback and without x. Enable feedback or supply x")
        elif not x is None:
            inputs = np.hstack((inputs, x))  # Add data inputs
        
        # Set parameters
        y_predicted = np.zeros(n_steps)
        
        # Get last states
        previous_y = self.y_last
        if not y_start is None:
            previous_y = self.normalize(outputs=y_start)[0]
        
        # Initialize state from last availble in train
        current_state = self.state[-1]
        
        # Predict iteratively
        for t in range(n_steps):
            # Get correct input based on feedback setting
            current_input = inputs[t] if not self.feedback else np.hstack((inputs[t], previous_y))
            
            # Update
            update = np.tanh(self.in_weights @ current_input.T + self.weights @ current_state)
            current_state = self.leaking_rate * update + (1 - self.leaking_rate) * current_state
            
            # Prediction. Order of concatenation is [1, inputs, y(n-1), state]
            complete_row = np.hstack((current_input, current_state))
            y_predicted[t] = complete_row @ self.out_weights
            previous_y = y_predicted[t]
        
        # Denormalize predictions
        y_predicted = self.denormalize(outputs=y_predicted)
        
        # Return predictions
        return y_predicted
    
    def predict_stepwise(self, y, x=None, steps_ahead=1, y_start=None):
        """Predicts a specified number of steps into the future for every time point in y-values array.
        
        E.g. if `steps_ahead` is 1 this produces a 1-step ahead prediction at every point in time.
        
        Parameters
        ----------
        y : numpy array
            Array with y-values. At every time point a prediction is made (excluding the current y)
        x : numpy array or None
            If prediciton requires inputs, provide them here
        steps_ahead : int (default 1)
            The number of steps to predict into the future at every time point
        y_start : float or None
            Starting value from which to start prediction. If None, last stored value from training will be used
        
        Returns
        -------
        y_predicted : numpy array
            Array of predictions at every time step of shape (times, steps_ahead)
        
        """
        # Check if ESN has been trained
        if self.out_weights is None or self.y_last is None:
            raise ValueError('Error: ESN not trained yet')
        
        # Normalize the arguments (like was done in train)
        y = self.normalize(outputs=y)
        if not x is None:
            x = self.normalize(inputs=x)
        
        # Timesteps in y
        t_steps = y.shape[0]

        # Choose correct input
        if x is None and not self.feedback:
            raise ValueError("Error: cannot run without feedback and without x. Enable feedback or supply x")
        elif not x is None:
            # Initialize input
            inputs = np.ones((t_steps, 1))  # Add bias term
            inputs = np.hstack((inputs, x))  # Add x inputs
        else:
            # x is None
            inputs = np.ones((t_steps + steps_ahead, 1))  # Add bias term
            
        # Run until we have no further inputs
        time_length = t_steps if x is None else t_steps - steps_ahead
        
        # Set parameters
        y_predicted = np.zeros((time_length, steps_ahead))
        
        # Get last states
        previous_y = self.y_last
        if not y_start is None:
            previous_y = self.normalize(outputs=y_start)[0]
        
        # Initialize state from last availble in train
        current_state = self.state[-1]
        
        # Predict iteratively
        for t in range(time_length):
            
            # State_buffer for steps ahead prediction
            prediction_state = np.copy(current_state)
            
            # Y buffer for step ahead prediction
            prediction_y = np.copy(previous_y)
            
            # Predict stepwise at from current time step
            for n in range(steps_ahead):
                
                # Get correct input based on feedback setting
                prediction_input = inputs[t + n] if not self.feedback else np.hstack((inputs[t + n], prediction_y))
                
                # Update
                prediction_update = np.tanh(self.in_weights @ prediction_input.T + self.weights @ prediction_state)
                prediction_state = self.leaking_rate * prediction_update + (1 - self.leaking_rate) * prediction_state
                
                # Store for next iteration of t (evolves true state)
                if n == 0:
                    current_state = np.copy(prediction_state)
                
                # Prediction. Order of concatenation is [1, inputs, y(n-1), state]
                prediction_row = np.hstack((prediction_input, prediction_state))
                y_predicted[t, n] = prediction_row @ self.out_weights
                prediction_y = y_predicted[t, n]
            
            # Evolve true state
            previous_y = y[t]
        
        # Denormalize predictions
        y_predicted = self.denormalize(outputs=y_predicted)
        
        # Return predictions
        return y_predicted
        
    def error(self, predicted, target, method='mse', alpha=1.):
        """Evaluates the error between predictions and target values.
        
        Parameters
        ----------
        predicted : array
            Predicted value
        target : array
            Target values
        method : {'mse', 'tanh', 'rmse', 'nmse', 'nrmse', 'tanh-nmse', 'log-tanh', 'log'}
            Evaluation metric. 'tanh' takes the hyperbolic tangent of mse to bound its domain to [0, 1] to ensure 
            continuity for unstable models. 'log' takes the logged mse, and 'log-tanh' takes the log of the squeezed
            normalized mse. The log ensures that any variance in the GP stays within bounds as errors go toward 0.
            
        alpha : float
            Alpha coefficient to scale the tanh error transformation: alpha * tanh{(1 / alpha) * error}
        
        Returns
        -------
        error : float
            The error as evaluated with the metric chosen above
        
        """      
        # Return error based on choices
        errors = predicted.ravel() - target.ravel()
        
        # Adjust for NaN and np.inf in predictions (unstable solution)
        # if not np.all(np.isfinite(predicted)):
        #     print("Warning: some predicted values are not finite")
        #     errors = np.inf
        
        if method == 'mse':
            error = np.mean(np.square(errors))
        elif method == 'tanh':
            error = alpha * np.tanh((1. / alpha) * np.mean(np.square(errors)))  # To 'squeeze' errors onto the interval (0, 1)
        elif method == 'rmse':
            error = np.sqrt(np.mean(np.square(errors)))
        elif method == 'nmse':
            error = np.mean(np.square(errors)) / np.square(target.ravel().std(ddof=1))
        elif method == 'nrmse':
            error = np.sqrt(np.mean(np.square(errors))) / target.ravel().std(ddof=1)
        elif method == 'tanh-nmse':
            nmse = np.mean(np.square(errors)) / np.square(target.ravel().std(ddof=1))
            error = alpha * np.tanh((1. / alpha) * nmse)
        elif method == 'log':
            mse = np.mean(np.square(errors))
            error = np.log(mse)
        elif method == 'log-tanh':
            nrmse = np.sqrt(np.mean(np.square(errors))) / target.ravel().std(ddof=1)
            error = np.log(alpha * np.tanh((1. / alpha) * nrmse))
        else:
            raise ValueError('Scoring method not recognized')
        return error

    
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
    scoring_method : {'mse', 'rmse', 'tanh', 'nmse', 'nrmse', 'log', 'log-tanh', 'nmse-tanh'}
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
                 max_iterations=1000, batch_size=1, cv_samples=1, mcmc_samples=None, scoring_method='tanh', 
                 tanh_alpha=1., esn_burn_in=100, acquisition_type='EI', max_time=np.inf, n_jobs=1, 
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
        
        # Temporarily store the data
        self.x = x
        self.y = y
        
        # Keywords to feed into Bayesian Optimization
        keyword_arguments = {'kernel': GPy.kern.Matern52(input_dim=7, ARD=True)}
        
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
        
        # Temporarily store the data
        self.x = x
        self.y = y
        
        # Inform user    
        if self.verbose:
            print("Model initialization and exploration run...")
        
        
        # Define objective
        objective = GPyOpt.core.task.SingleObjective(self.objective_sampler, 
                                                     objective_name = 'ESN Objective',
                                                     batch_type = 'synchronous',
                                                     num_cores=self.n_jobs)
        
        # Set search space and constraints (spectral radius - leaking rate ≤ 0)
        constraints = [{'name': 'alpha-rho', 'constrain': 'x[:, 3] - x[:, 2]'}]
        space = GPyOpt.core.task.space.Design_space(bounds, constraints)
        
        # Set GP kernel
        kernel = GPy.kern.Matern52(input_dim=7, ARD=True)
        gamma_prior = lambda: GPy.priors.Gamma(.001, .001)  # Proper distribution close to Jeffrey's prior
        kernel.variance.set_prior(gamma_prior())
        kernel.lengthscale.set_prior(gamma_prior())
        
        # Select model and acquisition
        if self.mcmc_samples is None:
            acquisition_type = self.acquisition_type
            model = GPyOpt.models.GPModel(kernel=kernel, 
                                          max_iters=1000,
                                          exact_feval=False, 
                                          normalize_Y=True, 
                                          optimizer='lbfgs', 
                                          optimize_restarts=1,
                                          verbose=self.verbose)
        else:
            acquisition_type = self.acquisition_type + '_MCMC'
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
        model.likelihood.variance.set_prior(gamma_prior())
        
        # Set acquisition TODO: Local Penalization
        acquisition_optimizer = GPyOpt.optimization.AcquisitionOptimizer(space, optimizer='lbfgs')
        SelectedAcquisition = GPyOpt.acquisitions.select_acquisition(acquisition_type)
        acquisition = SelectedAcquisition(model=model, space=space, optimizer=acquisition_optimizer)
        try:
            # Set jitter to 0 if used
            acquisition.jitter = 0.
        except AttributeError:
            pass
        
        # Set initial design
        initial_x = GPyOpt.util.stats.sample_initial_design('latin', space, self.initial_samples)  # Latin hypercube initialization
        
        # Pick evaluator
        evaluator = GPyOpt.core.evaluators.Predictive(acquisition=acquisition, batch_size=self.batch_size, normalize_Y=True)
        
        # Build optimizer
        update_interval = 5 if self.mcmc_samples is None else 25
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
        
        # Initialize new random state
        random_state = np.random.RandomState(self.seed + 2)
        
         # Get samples
        for i in range(self.cv_samples):  # TODO: Can be parallelized
            
            # Get indices
            start_index = random_state.randint(viable_start, viable_stop)
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
        
        # Return scores
        if self.verbose:    
            print('Objective scores:', scores)
        
        # Pass back as a column vector (as required by GPyOpt)
        mean_score = scores.mean().reshape(-1, 1) 
        return mean_score  
