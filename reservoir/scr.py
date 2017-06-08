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
        self.weights = np.zeros((self.n_nodes, self.n_nodes), dtype=float)
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
        """Generates states given some column vector x"""
        # Initialize new random state
        random_state = np.random.RandomState(self.seed)
        
        # Calculate correct shape
        rows = x.shape[0]
        
        # Build state matrix
        state = np.zeros((rows, self.n_nodes), dtype=float)
            
        # Set and scale input weights (for memory length and non-linearity)
        self.in_weights = np.full(shape=(self.n_nodes, x.shape[1]), fill_value=self.input_weight, dtype=float)
        self.in_weights *= np.sign(random_state.uniform(low=-1.0, high=1.0, size=self.in_weights.shape)) 
        
        # Set last state
        previous_state = state[0]
        
        # Train iteratively
        for t in range(rows):
            state[t] = np.tanh(self.in_weights @ x[t] + self.weights @ previous_state)
            previous_state = state[t]
        
        # Add intercept
        intercept = np.ones(shape=(rows, 1), dtype=state.dtype)
        state = np.hstack((intercept, state))
        
        return state[burn_in:]
    
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
        state, y, burn_in : tuple
            Returns the state, the y values provided and the number of time steps used for burn_in. 
            These data can be used for diagnostic purposes (e.g. vizualization of activations), or for
            cross-validation.
        
        """    
        state = self.generate_states(x, burn_in=burn_in)
        
        # Concatenate inputs with node states
        train_x = state
        train_y = y[burn_in:]  # Include everything after burn_in
        
        # Ridge regression
        ridge_x = train_x.T @ train_x + self.regularization * np.eye(train_x.shape[1], dtype=float)
        ridge_y = train_x.T @ train_y 
        
        # Solve for out weights
        try:
            # Cholesky solution (fast)
            self.out_weights = np.linalg.solve(ridge_x, ridge_y).reshape(-1, 1)
        except np.linalg.LinAlgError:
            # Pseudo-inverse solution
            self.out_weights = (scipy.linalg.pinvh(ridge_x) @ ridge_y).reshape(-1, 1)  # Robust solution if ridge_x is singular
        
        # Return all data for computation or visualization purposes
        return state, y, burn_in
        
    def validation_score(self, y, x, folds=5, scoring_method='L2', burn_in=30):
        """Trains and gives k-folds validation score"""
        # Get states
        state = self.generate_states(x, burn_in=burn_in)
                
        # Placeholder
        scores = np.zeros(folds, dtype=float)
        
        # Get y
        y = y[burn_in:]
        
        # Fold size
        fold_size = y.shape[0] // folds
        
        assert(fold_size > burn_in, 'Burn_in too large for current k-folds')
        
        for k in range(folds):
            
            # Validation folds
            start_index = k * fold_size
            stop_index = start_index + fold_size
            
            # Indices
            validation_indices = np.arange(start_index, stop_index)
            
            # Train mask
            train_mask = np.ones(y.shape[0], dtype=bool)
            train_mask[validation_indices] = False
            
            # Concatenate inputs with node states
            train_x = state[train_mask]
            train_y = y[train_mask]
            
            # Ridge regression
            ridge_x = train_x.T @ train_x + self.regularization * np.eye(train_x.shape[1])
            ridge_y = train_x.T @ train_y 
            
            # Solve for out weights
            try:
                # Cholesky solution (fast)
                out_weights = np.linalg.solve(ridge_x, ridge_y).reshape(-1, 1)
            except np.linalg.LinAlgError:
                # Pseudo-inverse solution
                out_weights = (scipy.linalg.pinvh(ridge_x) @ ridge_y).reshape(-1, 1)  # Robust solution if ridge_x is singular
            
            # Validation set
            validation_x = state[validation_indices]
            validation_y = y[validation_indices]
            
            # Predict
            prediction = validation_x @ out_weights
            
            # Score
            scores[k] = self.error(prediction, validation_y, scoring_method)
        
        # Return mean validation score
        return scores.mean()
        
    def train_validate_multiple(self, y, x, series_weights, folds=5, scoring_method='L2', burn_in=30):
        """Trains and gives k-folds validation score for multiple series
        
        Parameters
        ==========
        y : array
        x : array
        series_weights : array
            Array of n_series values, with the weighting of the series toward the regression
            
        Returns
        =======
        scores.mean() : float
            Mean error during cross-validation on weighted ridge regression
        
        """
        # Checks
        assert(y.shape == x.shape, 'Data matrices not of equal shape')
        
        # Easy retrieval
        t_steps = y.shape[0]
        n_series = y.shape[1]
        effective_length = t_steps - burn_in
        samples = effective_length * n_series
        
        # Concatenate all states
        states = np.zeros((samples, self.n_nodes + 1), dtype=float)  # Add one column for intercept
        all_y = np.zeros((samples, 1), dtype=float)
        for n, start_index in enumerate(range(0, samples, effective_length)):
            
            # Get states
            states[start_index: start_index + effective_length, :] = self.generate_states(x[:, n].reshape(-1, 1), burn_in=burn_in)
            
            # Concatenate output
            all_y[start_index: start_index + effective_length, 0] = y[burn_in:, n]
            
        # Series weights
        sample_weights = series_weights.repeat(effective_length)
        
        # Shuffle data
        random_state = np.random.RandomState(self.seed + 2)
        permutation = np.arange(samples)
        random_state.shuffle(permutation)
        shuffled_states = states[permutation]
        shuffled_y = all_y[permutation]
        shuffled_weights = sample_weights[permutation]
        
        # Placeholders
        scores = np.zeros(folds, dtype=float)
        out_weights = np.zeros((self.n_nodes + 1, folds), dtype=float)
        
        # Fold size
        fold_size = samples // folds
        
        # Checks
        assert(fold_size > burn_in, 'Burn-in too large for current k-folds cross-validation')
        
        # K-folds
        for k in range(folds):
            
            # Validation folds
            start_index = k * fold_size
            stop_index = start_index + fold_size
            
            # Indices
            validation_indices = np.arange(start_index, stop_index)
            
            # Train mask
            train_mask = np.ones(samples, dtype=bool)
            train_mask[validation_indices] = False
            
            # Concatenate inputs with node states
            train_x = shuffled_states[train_mask]
            train_y = shuffled_y[train_mask]
            masked_weights = shuffled_weights[train_mask]
            train_weights = np.diag(masked_weights)
                        
            # Ridge regression
            ridge_x = train_x.T @ train_weights @ train_x + self.regularization * np.eye(train_x.shape[1])
            ridge_y = train_x.T @ train_weights @ train_y 
            
            # Solve for out weights
            try:
                # Cholesky solution (fast)
                out_weights = np.linalg.solve(ridge_x, ridge_y).reshape(-1, 1)
            except np.linalg.LinAlgError:
                # Pseudo-inverse solution
                out_weights = (scipy.linalg.pinvh(ridge_x) @ ridge_y).reshape(-1, 1)  # Robust solution if ridge_x is singular
            
            # Validation set
            validation_x = shuffled_states[validation_indices]
            validation_y = shuffled_y[validation_indices]
            
            # Predict
            prediction = validation_x @ out_weights
            
            # # Save
            # scores[k] = self.error(prediction, validation_y, scoring_method)
            out_weights[:, k] = out_weights
        
        # Return mean validation score
        self.out_weights = out_weights.mean(axis=1)
        # return scores.mean()
            
    def test(self, y, x, out_weights=None, scoring_method='L2', burn_in=30, alpha=1., **kwargs):
        """Tests and scores against known output.
        
        Parameters
        ----------
        y : array
            Column vector of known outputs
        x : array or None
            Any inputs if required
        scoring_method : {'L2', 'mse', 'rmse', 'nrmse', 'tanh'}
            Evaluation metric used to calculate error
        burn_in : int
            Number of time steps to exclude from prediction initially
        alpha : float
            Alpha coefficient to scale the tanh error transformation: alpha * tanh{(1 / alpha) * error}
            
        Returns
        -------
        error : float
            Error between prediction and known outputs
        
        """
        # Run prediction
        y_predicted = self.predict_stepwise(x, out_weights=out_weights)
        
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
            
        # Return predictions
        return y_predicted
        
    def error(self, predicted, target, method='L2', alpha=1.):
        """Evaluates the error between predictions and target values.
        
        Parameters
        ----------
        predicted : array
            Predicted value
        target : array
            Target values
        method : {'L2', 'mse', 'tanh', 'rmse', 'nmse', 'nrmse', 'tanh-nmse', 'log-tanh', 'log'}
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
        errors = target.ravel() - predicted.ravel()
        
        # Adjust for NaN and np.inf in predictions (unstable solution)
        if not np.all(np.isfinite(predicted)):
            # print("Warning: some predicted values are not finite")
            errors = np.inf
        
        # Compute mean error
        if method == 'L2':
            error = np.linalg.norm(errors, ord=2)  # Technically the same as rmse
        elif method == 'mse':
            error = np.mean(np.square(errors))
        elif method == 'tanh':
            error = alpha * np.tanh(np.mean(np.square(errors)) / alpha)  # To 'squeeze' mse onto the interval (0, 1)
        elif method == 'rmse':
            error = np.sqrt(np.mean(np.square(errors)))
        elif method == 'nmse':
            error = np.mean(np.square(errors)) / np.square(target.ravel().std(ddof=1))
        elif method == 'nrmse':
            error = np.sqrt(np.mean(np.square(errors))) / target.ravel().std(ddof=1)
        elif method == 'tanh-nmse':
            nmse = np.mean(np.square(errors)) / np.square(target.ravel().std(ddof=1))
            error = alpha * np.tanh(nmse / alpha)
        else:
            raise ValueError('Scoring method not recognized')
        return error
