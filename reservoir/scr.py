import numpy as np
import scipy


__all__ = ['SimpleCycleReservoir']


class SimpleCycleReservoir:
    
    def __init__(self, n_nodes=30, regularization=1e-8, cyclic_weight=0.5, input_weight=0.5, random_seed=123):
        # Save attributes
        self.n_nodes = int(np.round(n_nodes))
        self.regularization = regularization
        self.cyclic_weight = cyclic_weight
        self.input_weight = input_weight
        self.seed = random_seed
        
        # Generate reservoir
        self.generate_reservoir()
        
    def generate_reservoir(self):
        """Generates transition weights"""
        # Set reservoir weights
        self.weights = np.zeros((self.n_nodes, self.n_nodes))
        self.weights[0, -1] = self.cyclic_weight
        for i in range(self.n_nodes - 1):
            self.weights[i+1, i] = self.cyclic_weight
        
        # Set out to none to indicate untrained ESN
        self.out_weights = None
    
    def draw_reservoir(self):
        """Vizualizes reservoir. 
        
        Note: Requires 'networkx' package.
        
        """
        import networkx as nx
        graph = nx.DiGraph(self.weights)
        nx.draw(graph)
        
    def generate_states(self, x, burn_in=30):
        """Generates states given some input x"""
        # Initialize new random state
        random_state = np.random.RandomState(self.seed)
            
        # Normalize inputs and outputs
        # y = self.normalize(outputs=y, keep=True)
        # x = self.normalize(inputs=x, keep=True)
        
        # Calculate correct shape
        rows = x.shape[0]
        
        # Build state matrix
        state = np.zeros((rows, 1 + self.n_nodes))
        state[:, 0] = np.ones(shape=state.shape[0], dtype=state.dtype)  # Add intercept
            
        # Set and scale input weights (for memory length and non-linearity)
        self.in_weights = np.full(shape=(x.shape[1], self.n_nodes), fill_value=self.input_weight, dtype=float)
        self.in_weights *= np.sign(random_state.uniform(low=-1.0, high=1.0, size=self.in_weights.shape)) 
        
        # Set last state
        previous_state = state[0]
        
        # Train iteratively
        for t in range(rows):
            state[t, 1:] = np.tanh(x[t] @ self.in_weights + previous_state @ self.weights)
            previous_state = state[t]
        
        return state[burn_in:]
        
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
    
    def train(self, y, x, burn_in=30):
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
            Returns the complete dataset (state matrix concatenated with any inputs),
            the y values provided and the number of time steps used for burn_in. These data can be used
            for diagnostic purposes  (e.g. vizualization of activations).
        
        """    
        state = self.generate_states(x, burn_in=burn_in)
        
        # Concatenate inputs with node states
        train_x = state  # Add intercept
        train_y = y[burn_in:]  # Include everything after burn_in
        
        # Ridge regression
        ridge_x = train_x.T @ train_x + self.regularization * np.eye(train_x.shape[1])
        ridge_y = train_x.T @ train_y 
        
        # Solve for out weights
        try:
            # Cholesky solution (fast)
            self.out_weights = np.linalg.solve(ridge_x, ridge_y).reshape(-1, 1)
        except np.linalg.LinAlgError:
            # Pseudo-inverse solution
            self.out_weights = scipy.linalg.pinvh(ridge_x, ridge_y, rcond=1e6*np.finfo('d').eps).reshape(-1, 1)  # Robust solution if ridge_x is singular
        
        # Return all data for computation or visualization purposes (Note: these are normalized)
        return state, y, burn_in
            
    def test(self, y, x=None, scoring_method='mse', alpha=1., burn_in=30, **kwargs):
        """Tests and scores against known output.
        
        Parameters
        ----------
        y : array
            Column vector of known outputs
        x : array or None
            Any inputs if required
        scoring_method : {'mse', 'rmse', 'nrmse', 'tanh'}
            Evaluation metric used to calculate error
        burn_in : int
            Number of time steps to exclude from prediction initially
        alpha : float
            Alpha coefficient to scale the tanh error transformation: alpha * tanh{(1 / alpha) * error}
            
        Returns
        -------
        error : float
            Error between prediction and knwon outputs
        
        """
        # Run prediction
        final_t = y.shape[0]
        y_predicted = self.predict_stepwise(x)
        
        # Checks
        assert(y_predicted.shape[0] == y.shape[0])
            
        # Return error
        return self.error(y_predicted[burn_in:], y[burn_in:], scoring_method, alpha=alpha)
    
    def predict_stepwise(self, x, out_weights=None, **kwargs):
        """Predicts a specified number of steps into the future for every time point in y-values array.
        
        Parameters
        ----------
        x : numpy array or None
            If prediciton requires inputs, provide them here. If y has T time samples, x should have at least T + N - 1,
            time samples for N step ahead prediction, otherwise some step ahead predictions may be undefined (NaN)
        out_weights : numpy array (2D column vector)
            The weights to use for prediction. Overrides any trained weights stored on the object.
        
        Returns
        -------
        y_predicted : numpy array
            Array of predictions at every time step of shape (times, steps_ahead)
        
        """
        # Check if ESN has been trained
        if self.out_weights is None and out_weights is None:
            raise ValueError('Error: Train model or provide out_weights')
        
        # Get states
        state = self.generate_states(x, burn_in=0)
        
        # Select weights
        if not out_weights is None:
            weights = out_weights  # Provided
        else:
            weights = self.out_weights  # From training
        
        # Predict
        y_predicted = state @ weights
            
        # Denormalize predictions
        # y_predicted = self.denormalize(outputs=y_predicted)
            
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
            Alpha coefficient to scale the tanh error transformation: alpha * tanh{(1 / alpha) * error}.
            This squeezes errors onto the interval [0, alpha].
            Default is 1. Suggestions for squeezing errors > n * stddev of the original series 
            (for tanh-nrmse, this is the point after which difference with y = x is larger than 50%,
             and squeezing kicks in):
            
        Returns
        -------
        error : float
            The error as evaluated with the metric chosen above
        
        """      
        errors = predicted.ravel() - target.ravel()
        
        # Adjust for NaN and np.inf in predictions (unstable solution)
        if not np.all(np.isfinite(predicted)):
            # print("Warning: some predicted values are not finite")
            errors = np.inf
        
        # Compute mean error
        if method == 'mse':
            error = np.mean(np.square(errors))
        elif method == 'tanh':
            error = alpha * np.tanh(np.mean(np.square(errors)) / alpha)  # To 'squeeze' errors onto the interval (0, 1)
        elif method == 'rmse':
            error = np.sqrt(np.mean(np.square(errors)))
        elif method == 'nmse':
            error = np.mean(np.square(errors)) / np.square(target.ravel().std(ddof=1))
        elif method == 'nrmse':
            error = np.sqrt(np.mean(np.square(errors))) / target.ravel().std(ddof=1)
        elif method == 'tanh-nrmse':
            nrmse = np.sqrt(np.mean(np.square(errors))) / target.ravel().std(ddof=1)
            error = alpha * np.tanh(nrmse / alpha)
        elif method == 'log':
            mse = np.mean(np.square(errors))
            error = np.log(mse)
        elif method == 'log-tanh':
            nrmse = np.sqrt(np.mean(np.square(errors))) / target.ravel().std(ddof=1)
            error = np.log(alpha * np.tanh((1. / alpha) * nrmse))
        else:
            raise ValueError('Scoring method not recognized')
        return error
