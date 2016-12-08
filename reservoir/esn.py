import numpy as np
import scipy.stats
import scipy.linalg
import GPy
import GPyOpt


__all__ = ['EchoStateNetwork', 'EchoStateNetworkCV']


class EchoStateNetwork:
    """Builds and echo state network with the specified parameters.
    In training, testing and predicting, x is a matrix consisting of column-wise time series features. 
    Y is a zero-dimensional target vector.
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
        """Generates random reservoir from parameters set at initialization."""
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
        """Visulaizes reservoir."""
        import networkx as nx
        graph = nx.DiGraph(self.weights)
        nx.draw(graph)
        
    def normalize(self, inputs=None, outputs=None, store=False):
        """Normalizes array by column (along rows) and stores mean and standard devation.
        Set store to True if you want to retain means and stds for denormalization later"""      
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
        """Denormalizes array by column (along rows) using stored mean and standard deviation"""
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
    
    def train(self, x=None, y=None, burn_in=100):
        """Trains the out weights on the random network. y is required input. 
        Running a burn-in of a sizable length. This makes sure the state 
        matrix has converged to a 'credible' value.
        """
        # Checks
        if y is None:
            raise ValueError("Error: need to supply y")
        
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
            
    def test(self, x=None, y=None, y_start=None, scoring_method='mse'):
        """Tests and scores against known output."""
        # Checks
        if y is None:
            raise ValueError('Error: need to supply at least a target y to compare to.')

        # Run prediction
        y_predicted = self.predict(y.shape[0], x, y_start=y_start)
        
        # Return error
        return self.error(y_predicted, y, scoring_method)
    
    def predict(self, n_steps, x=None, y_start=None):
        """Predicts n values in advance, starting from the last state generated in training."""
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
        """
        Calculates the root mean squared error.
        Method can also be 'rmse'.
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
        elif method == 'negative_logposterior':
            n = errors.shape[0]
            sse = np.square(errors).sum()
            ssw = self.out_weights.T @ self.out_weights
            var = np.square(errors.std(ddof=1))
            ols_loglikelihood = (- (n / 2) * np.log(2 * np.pi) - (n / 2) * np.log(var) - (1 / (2*var)) * sse).ravel()
            ridge_logprior = (- (self.regularization / 2) * ssw).ravel()
            logposterior = ols_loglikelihood + ridge_logprior
            logposterior = logposterior if np.isfinite(logposterior) else -np.inf
            return -logposterior
        else:
            raise ValueError('Scoring method not recognized')
        return error

    
class EchoStateNetworkCV:
    """A cross-validation object that optimizes ESN hyperparameters. Takes subsamples within bounds."""
    
    def __init__(self, bounds, subsequence_length, initial_samples=8, validate_fraction=0.2, iterations=10, 
                 batch_size=8, cv_samples=1, mcmc_samples=None, scoring_method='tanh', esn_burn_in=100, 
                 max_time=np.inf, n_jobs=8, verbose=True):
        self.bounds = bounds
        self.subsequence_length = subsequence_length
        self.initial_samples = initial_samples
        self.validate_fraction = validate_fraction
        self.iterations = iterations
        self.batch_size = batch_size  # GPyOPt does currently not support this well
        self.cv_samples = cv_samples
        self.mcmc_samples = mcmc_samples
        self.scoring_method = scoring_method
        self.esn_burn_in = esn_burn_in
        self.max_time = max_time
        self.n_jobs = n_jobs
        self.verbose = verbose
    
    def optimize(self, y, x=None):
        """Uses Bayesian Optimization with Gaussian Process priors to optimize ESN hyperparameters.
        Returns the best parameters model."""
        
        # Temporarily store the data
        self.x = x
        self.y = y
        
        # Keyword to feed into Bayesian Optimization
        keyword_arguments = {'kernel': GPy.kern.sde_Matern52(input_dim=7, ARD=True)}
        
        if self.mcmc_samples is None:
            # MLE solution
            model_type = 'GP'
            acquisition_type = 'LCB'
        else:
            # MCMC solution
            model_type = 'GP_MCMC'
            acquisition_type = 'LCB_MCMC'
            keyword_arguments['n_samples'] = self.mcmc_samples
            
        # Set contraint (spectral radius - leaking rate ≤ 0)
        constraints = [{'name': 'alpha-rho', 'constrain': 'x[:, 3] - x[:, 2]'}]
        
        if self.verbose:
            print("Model initialization and exploration run...")
        
        # Build optimizer    
        self.optimizer = GPyOpt.methods.BayesianOptimization(self.objective_sampler,
                                                             domain=self.bounds,
                                                             initial_design_numdata=self.initial_samples,
                                                             constrains=constraints, 
                                                             model_type=model_type, 
                                                             acquisition_type=acquisition_type, 
                                                             acquisition_optimizer_type = 'CMA',
                                                             normalize_Y=True, 
                                                             evaluator_type='local_penalization', 
                                                             #num_cores=self.n_jobs, 
                                                             #batch_size=self.batch_size,
                                                             **keyword_arguments)
        # Show model
        if self.verbose:
            print(self.optimizer.model.model, '\n')
        
        if self.verbose:
            print("Starting optimization...")
        
        # Optimize
        self.optimizer.run_optimization(max_iter=self.iterations, max_time=self.max_time)
        
        # Clean up memory (for Jupyter Notebook)
        self.x = None
        self.y = None
        del self.x
        del self.y
        
        if self.verbose:        
            print('Done.')
        
        # Show convergence
        self.optimizer.plot_convergence()
        
        # Return best parameters
        best_found = self.optimizer.x_opt.T
        arguments = dict(input_scaling=best_found[0], feedback_scaling=best_found[1], leaking_rate=best_found[2], 
                         spectral_radius=best_found[3], regularization=10.**best_found[4], 
                         connectivity=10.**best_found[5], n_nodes=best_found[6], feedback=True)
        return arguments
        
    def objective_function(self, parameters, train_y, validate_y, train_x=None, validate_x=None):
        """Returns selected error metric on validation set. Parameters is a column vector shape: (n, 1)"""
        # Get arguments
        arguments = parameters.T

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
        """Splits training set into train and validate sets, and computes multiple runs of the objective function"""
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
            validate_length = np.round(self.subsequence_length * self.validate_fraction).astype(int)
            train_length = self.subsequence_length - validate_length
            
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
        return np.mean(scores).reshape(-1, 1)  # Pass back as column of scores
        
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
