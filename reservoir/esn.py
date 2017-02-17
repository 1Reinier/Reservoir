import numpy as np
import scipy.stats
import scipy.linalg
import GPy
import GPyOpt
import copy
import json


__all__ = ['EchoStateNetwork', 'EchoStateNetworkCV']


class EchoStateNetwork:
    """EchoStateNetwork(n_nodes, input_scaling=0.5, feedback_scaling=0.5, spectral_radius=0.8, 
                        leaking_rate=1.0, connectivity=0.1, regularization=1e-8, feedback=True)
    
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
    
    """
    
    def __init__(self, n_nodes, input_scaling=0.5, feedback_scaling=0.5, spectral_radius=0.8, 
                 leaking_rate=1.0, connectivity=0.1, regularization=1e-8, feedback=True):
        # Parameters
        self.n_nodes = int(np.round(n_nodes))
        self.input_scaling = input_scaling
        self.feedback_scaling = feedback_scaling
        #self.noise_scaling = noise_scaling  # Ridge is enough?
        self.spectral_radius = spectral_radius
        self.connectivity = connectivity
        self.input_scaling = input_scaling
        self.leaking_rate = leaking_rate
        self.regularization = regularization
        self.feedback = feedback
        self.generate_reservoir()
        
    def generate_reservoir(self):
        """generate_reservoir(self)
        
        Generates random reservoir from parameters set at initialization.
        
        """
        # Set weights and sparsity randomly
        self.weights = np.random.uniform(-1., 1., size=(self.n_nodes, self.n_nodes))
        accept = np.random.uniform(size=(self.n_nodes, self.n_nodes)) < self.connectivity
        self.weights *= accept
    
        # Set spectral density
        max_eigenvalue = np.abs(np.linalg.eigvals(self.weights)).max()  # Any == -1 -> look up
        self.weights *= self.spectral_radius / max_eigenvalue
        
        # Default state
        self.state = np.zeros((1, self.n_nodes))
        
        # Set out to none to indicate untrained ESN
        self.out_weights = None
        
    def draw_reservoir(self):
        """draw_reservoir(self)
        
        Vizualizes reservoir. Requires 'networkx' package.
        
        """
        import networkx as nx
        graph = nx.DiGraph(self.weights)
        nx.draw(graph)
        
    def normalize(self, inputs=None, outputs=None, store=False):
        """normalize(self, inputs=None, outputs=None, store=False)
        
        Normalizes array by column (along rows) and stores mean and standard devation.
        Set `store` to True if you want to retain means and stds for denormalization later.
        
        Parameters
        ----------
        inputs : array or None
            Input matrix that is to be normalized
        outputs : array or None
            Output column vector that is to be normalized
        store : bool
            Stores the normalization transformation in the object to denormalize later
        
        Returns
        -------
        transformed : tuple or array
            Returns tuple of every normalized array. In case only one object is to be returned the tuple will be 
            unpacked before returning
        
        """      
        if inputs is None and outputs is None:
            raise ValueError('Inputs and outputs cannot both be None')
        
        # Storage for transformed variables
        transformed = []
        
        if not inputs is None:
            if store:
                # Store for denormalization
                self._input_means = inputs.mean(axis=0)
                self._input_stds = inputs.std(ddof=1, axis=0)
                
                # Do not normalize bias
                self._input_means[0] = 0
                self._input_stds[0] = 1
                
            # Transform
            transformed.append((inputs - self._input_means) / self._input_stds)
            
        if not outputs is None:
            if store:
                # Store for denormalization
                self._output_means = outputs.mean(axis=0)
                self._output_stds = outputs.std(ddof=1, axis=0)

            # Transform
            transformed.append((outputs - self._output_means) / self._output_stds)
        
        # Syntactic sugar
        return tuple(transformed) if len(transformed) > 1 else transformed[0]
        
    def denormalize(self, inputs=None, outputs=None):
        """denormalize(self, inputs=None, outputs=None)
        
        Denormalizes array by column (along rows) using stored mean and standard deviation.
        
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
        """train(self, y, x=None, burn_in=100
        
        Trains the out weights on the random network. This is needed before being able to make predictions.
        Consider running a burn-in of a sizable length. This makes sure the state  matrix has converged to a 
        'credible' value.
        
        Parameters
        ----------
        y : array
            Column vector of y values
        x : array or None
            Matrix of inputs (optional)
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
        
        # Reset state
        current_state = self.state[-1]  # From default or pretrained state
        
        # Calculate correct shape based on feedback (one row less)
        rows = y.shape[0] - 1 if self.feedback else y.shape[0]
        start_index = 1 if self.feedback else 0  # Convenience index
        
        # Build state matrix
        self.state = np.zeros((rows, self.n_nodes))
        
        # Build inputs
        inputs = np.ones((rows, 1))  # Add bias for all t = 0, ..., T
        
        # Add data inputs if present
        if x is not None:
            inputs = np.hstack((inputs, x[start_index:]))  # Add data inputs
            
        # Set and scale input weights (for memory length and non-linearity)
        self.in_weights = self.input_scaling * np.random.uniform(-1, 1, size=(self.n_nodes, inputs.shape[1]))
                
        # Add feedback if requested, optionally with feedback scaling
        if self.feedback:
            inputs = np.hstack((inputs, y[:-1]))  # Add teacher forced signal (equivalent to y(t-1) as input)
            feedback_weights = self.feedback_scaling * np.random.uniform(-1, 1, size=(self.n_nodes, 1))
            self.in_weights = np.hstack((self.in_weights, feedback_weights))
            
        # Normalize inputs and outputs
        inputs, y = self.normalize(inputs, y, store=True)
                
        # Train iteratively
        for t in range(inputs.shape[0]):
            update = np.tanh(self.in_weights @ inputs[t].T + self.weights @ current_state)
            current_state = self.leaking_rate * update + (1 - self.leaking_rate) * current_state  # Leaking separate
            self.state[t] = current_state
        
        # Concatenate inputs with node states
        complete_data = np.hstack((inputs, self.state))
        train_x = complete_data[burn_in:]  # Include everything after burn_in
        train_y = y[burn_in + 1:] if self.feedback else y[burn_in:]
                
        # Ridge regression, pseudo-inverse solution
#         ridge_x = np.vstack((train_x, np.sqrt(self.regularization) * np.eye(train_x.shape[1])))
#         ridge_y = np.vstack((train_y, np.zeros((train_x.shape[1], 1))))
#         self.out_weights = np.linalg.pinv(ridge_x) @ ridge_y
        
        # Ridge regression, full inverse solution
        self.out_weights = np.linalg.inv(train_x.T @ train_x + self.regularization * \
                                        np.eye(train_x.shape[1])) @ train_x.T @ train_y 

        # Store last y value as starting value for predictions
        self.y_last = y[-1]
        
        # Return all data for computation or visualization purposes (Note: these are normalized)
        return complete_data, (y[1:] if self.feedback else y), burn_in
            
    def test(self, y, x=None, y_start=None, scoring_method='mse'):
        """test(self, y, x=None, y_start=None, scoring_method='mse')
        
        Tests and scores against known output.
        
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
            
        Returns
        -------
        error : float
            Error between prediction and knwon outputs
        
        """
        # Run prediction
        y_predicted = self.predict(y.shape[0], x, y_start=y_start)
        
        # Return error
        return self.error(y_predicted, y, scoring_method)
    
    def predict(self, n_steps, x=None, y_start=None):
        """predict(self, n_steps, x=None, y_start=None)
        
        Predicts n values in advance, starting from the last state generated in training.
        
        Parameters
        ----------
        n_steps : int
            The number of steps to predict into the future (internally done in one step increments)
        x : array or None
            If prediciton requires inputs, provide them here
        y_start : float or None
            Starting value from which to start prediction. If None, last stored value from trainging will be used
        
        """
        # Check if ESN has been trained
        if self.out_weights is None or self.y_last is None:
            raise ValueError('Error: ESN not trained yet')
        
        # Initialize input
        inputs = np.ones((n_steps, 1)) # Add bias term
        
        # Choose correct input
        if x is None and not self.feedback:
            raise ValueError("Error: cannot run without feedback and without x. Enable feedback or supply x")

        elif x is not None:
            inputs = np.hstack((inputs, x))  # Add data inputs
        
        # Set parameters
        y_predicted = np.zeros(n_steps)
        
        # Get last states
        previous_y = self.y_last
        if y_start:
            previous_y = self.normalize(outputs=y_start)[0]
        
        # Initialize state from last availble in train
        current_state = self.state[-1]
        
        # Normalize the inputs (as is done in train)
        inputs = self.normalize(inputs)
        
        # Exclude last column if feedback is enabled (since feedback is calculated on the fly below)
        if self.feedback:
            inputs = inputs[:, :-1]
        
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
        
    def error(self, predicted, target, method='mse'):
        """error(self, predicted, target, method='mse')
        
        Evaluates the error between predictions and target values.
        
        Parameters
        ----------
        predicted : array
            Predicted value
        target : array
            Target values
        method : {'mse', 'tanh', 'rmse', 'nrmse'}
            Evaluation metric. 'tanh' takes the hyperbolic tangent of mse to bound its domain to [0, 1]
        
        Returns
        -------
        error : float
            The error as evaluated with the metric chosen above
        
        """      
        # Return error based on choices
        errors = predicted.ravel() - target.ravel()
        
        # Adjust for NaN and np.inf in predictions (unstable solution)
        if not np.all(np.isfinite(predicted)):
            #print("Warning: some predicted values are not finite")
            errors = np.inf
        
        if method == 'mse':
            error = np.mean(np.square(errors))
        elif method == 'tanh':
            error = np.tanh(np.mean(np.square(errors)))  # To 'squeeze' errors onto the interval (0, 1)
        elif method == 'rmse':
            error = np.sqrt(np.mean(np.square(errors)))
        elif method == 'nrmse':
            error = np.sqrt(np.mean(np.square(errors))) / target.ravel().std(ddof=1)
        # elif method == 'negative_logposterior':
        #     n = errors.shape[0]
        #     sse = np.square(errors).sum()
        #     ssw = self.out_weights.T @ self.out_weights
        #     var = np.square(errors.std(ddof=1))
        #     ols_loglikelihood = (- (n / 2) * np.log(2 * np.pi) - (n / 2) * np.log(var) - (1 / (2*var)) * sse).ravel()
        #     ridge_logprior = (- (self.regularization / 2) * ssw).ravel()
        #     logposterior = ols_loglikelihood + ridge_logprior
        #     logposterior = logposterior if np.isfinite(logposterior) else -np.inf
        #     return -logposterior
        else:
            raise ValueError('Scoring method not recognized')
        return error

    
class EchoStateNetworkCV:
    """EchoStateNetworkCV(bounds, subsequence_length, eps=1e-8, initial_samples=8, validate_fraction=0.2, 
                          max_iterations=1000, batch_size=4, cv_samples=1, mcmc_samples=None, scoring_method='tanh', 
                          esn_burn_in=100, max_time=np.inf, n_jobs=4, verbose=True)
    
    A cross-validation object that automatically optimizes ESN hyperparameters using Bayesian optimization with
    Gaussian Process priors. Tries to find optimal solution within the provided bounds.
    
    Parameters
    ----------
    bounds : list
        List of dicts specifying the bounds for optimization. Every dict has to contain entries for the following keys:
        name : string
        type : {'continuous', 'discrete'}
        domain : tuple
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
    scoring_method : {'mse', 'rmse', 'tanh'}
        Evaluation metric that is used to guide optimization
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
                 esn_burn_in=100, acquisition_type='EI', max_time=np.inf, n_jobs=1, verbose=True):
        self.bounds = bounds
        self.subsequence_length = subsequence_length
        self.eps = eps
        self.initial_samples = initial_samples
        self.validate_fraction = validate_fraction
        self.max_iterations = max_iterations
        self.batch_size = batch_size  # GPyOPt does currently not support this well; Currently ignored
        self.cv_samples = cv_samples
        self.mcmc_samples = mcmc_samples
        self.scoring_method = scoring_method
        self.esn_burn_in = esn_burn_in
        self.acquisition_type = acquisition_type
        self.max_time = max_time
        self.n_jobs = n_jobs  # Currently ignored
        self.verbose = verbose
        
        # Normalize bounds domains and remember transformation
        self.scaled_bounds, self.bound_scalings, self.bound_intercepts = self.normalize_bounds(self.bounds)
            
    def normalize_bounds(self, bounds):
        """normalize_bounds(self, bounds)
        
        Makes sure all bounds feeded into GPyOpt are scaled to the domain [0, 1], 
        to aid interpretation of convergence plots. Scalings are saved in instance parameters.
        
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
    
    def denormalize_arguments(self, normalized_arguments):
        """denormalize_arguments(self, model_arguments)
        
        Denormalize arguments to feed into model.
        
        Parameters
        ----------
        normalized_arguments : 1-D numpy array
            Contains arguments in same order as bounds
            
        Returns
        -------
        denormalized_arguments : 1-D numpy array
            Array with denormalized arguments
        
        """
        denormalized_arguments = (normalized_arguments * self.bound_scalings) + self.bound_intercepts
        return denormalized_arguments
    
    def optimize(self, y, x=None, store_path=None):
        """optimize(self, y, x=None)
        
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
        # Temporarily store the data
        self.x = x
        self.y = y
        
        # Keyword to feed into Bayesian Optimization
        keyword_arguments = {'kernel': GPy.kern.sde_Matern52(input_dim=7, ARD=True)}
        
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
            keyword_arguments['evaluator_type'] = 'local_penalization'
            
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
                                                             #normalize_Y=True,
                                                             acquisition_optimizer_type='lbfgs',
                                                             verbosity=self.verbose,
                                                             num_cores=self.n_jobs, 
                                                             batch_size=self.batch_size,
                                                             **keyword_arguments
                                                             ) 
                                                             
                                                             
        # Show model
        if self.verbose:
            print("Model initialization done.", '\n')
            print(self.optimizer.model.model, '\n')
        
        if self.verbose:
            print("Starting optimization...")
        
        # Optimize
        self.optimizer.run_optimization(eps=self.eps, max_iter=self.max_iterations, max_time=self.max_time, 
                                        verbosity=self.verbose)
        
        # Clean up memory (for Jupyter Notebook)
        self.x = None
        self.y = None
        del self.x
        del self.y
        
        if self.verbose:        
            print('Done.')
        
        # Show convergence
        self.optimizer.plot_convergence()
        
        # Scale arguments
        best_found = self.denormalize_arguments(self.optimizer.x_opt).T
        
        # Store in dict
        best_arguments = dict(input_scaling=best_found[0], feedback_scaling=best_found[1], leaking_rate=best_found[2], 
                         spectral_radius=best_found[3], regularization=10.**best_found[4], 
                         connectivity=10.**best_found[5], n_nodes=best_found[6], feedback=True)
        
        # Save to disk if desired
        if not store_path is None:
            with open(store_path, 'w+') as output_file:
                json.dump(best_arguments, output_file, indent=4)
        
        # Return best parameters
        return best_arguments
        
    def objective_function(self, parameters, train_y, validate_y, train_x=None, validate_x=None):
        """objective_function(self, parameters, train_y, validate_y, train_x=None, validate_x=None)
        
        Returns selected error metric on validation set. Parameters is a column vector shape: (n, 1).
        
        Parameters
        ----------
        parameters : array
            Parametrization of the Echo State Network
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
        arguments = self.denormalize_arguments(parameters).T

        # Build network
        esn = EchoStateNetwork(input_scaling=arguments[0], feedback_scaling=arguments[1], leaking_rate=arguments[2], 
                               spectral_radius=arguments[3], regularization=10.**arguments[4], 
                               connectivity=10.**arguments[5], n_nodes=arguments[6], feedback=True)
        # Train
        esn.train(x=train_x, y=train_y, burn_in=self.esn_burn_in)

        # Validation score
        score = esn.test(x=validate_x, y=validate_y, scoring_method=self.scoring_method)
        return score

    def objective_sampler(self, parameters):
        """objective_sampler(self, parameters)
        
        Splits training set into train and validate sets, and computes multiple samples of the objective function.
        
        Parameters
        ----------
        parameters : array
            Parametrization of the Echo State Network
        
        Returns
        -------
        mean_score : 2-D array
            Column vector with mean score(s), as required by GPyOpt
        
        """
        training_y = self.y
        training_x = self.x
        
        # Set viable sample range
        viable_start = self.esn_burn_in
        viable_stop = training_y.shape[0] - self.subsequence_length
        
        # Get sample lengths
        validate_length = np.round(self.subsequence_length * self.validate_fraction).astype(int)
        train_length = self.subsequence_length - validate_length
        
        # Score storage
        scores = np.zeros(self.cv_samples)
        
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
            
            cv_score = self.objective_function(parameters, train_y, validate_y, train_x, validate_x)
            scores[i] = cv_score
        
        # Return scores
        if self.verbose:    
            print('Objective scores:', scores)
        
        # Pass back as column of scores
        mean_score = np.mean(scores).reshape(-1, 1) 
        return mean_score  
        
#         # Parallel execution of models
#         tasks_left = copy.deepcopy(self.cv_samples)
        
#         while tasks_left > 0:
            
#             # Spawn parallel pool
#             with Pool(processes=min(self.n_jobs, tasks_left)) as pool:
                
#                 # Results container
#                 results = []
                
#                 # Set subsequences
#                 for n in range(self.n_jobs):
                    
#                     # Get indices
#                     start_index = np.random.randint(viable_start, viable_stop)
#                     train_stop_index = start_index + train_length
#                     validate_stop_index = train_stop_index + validate_length

#                     # Get samples
#                     train_y = training_y[start_index: train_stop_index]
#                     validate_y = training_y[train_stop_index: validate_stop_index]

#                     if not training_x is None:
#                         train_x = training_x[start_index: train_stop_index]
#                         validate_x = training_x[train_stop_index: validate_stop_index]
#                     else:
#                         train_x = None
#                         validate_x = None
                
#                     # Consolidate arguments
#                     arguments = (parameters, train_y, validate_y, train_x, validate_x)
#                     results.append(arguments)
                
#                 # Parallel dispatch 
#                 results = map(lambda parameters: pool.apply_async(self.objective_function, parameters), results)
            
#                 # Get and store results
#                 results = map(lambda result: result.get(), results)
#                 scores += results
                
#             # Update tasks left
#             tasks_left -= self.n_jobs
