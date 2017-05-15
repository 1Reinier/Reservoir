import numpy as np
from .esn_cv import *
from .scr import *


__all__ = ['ClusteringBO']


class ClusteringBO(EchoStateNetworkCV):
    """Bayesian Optimization with an objective optimized for ESN Clustering (Gianniotis 2017)
    
    Parameters
    ----------
    bounds : dict
        A dictionary specifying the bounds for optimization. The key is the parameter name and the value 
        is a tuple with minimum value and maximum value of that parameter. E.g. {'n_nodes': (100, 200), ...}
    readouts : array
        k-column matrix, where k is the number of clusters
    responsibilities : array
        matrix of shape (n, k) that contains membership probabilities for every series to every cluster
    
    """
    
    def __init__(self, bounds, responsibilities, readouts=None, eps=1e-6, initial_samples=100, max_iterations=300, log_space=True,
                 burn_in=30, seed=123, verbose=True, **kwargs):
        
        # Initialize optimizer
        super().__init__(bounds, subsequence_length=-1, model=SimpleCycleReservoir, eps=eps, 
                         initial_samples=initial_samples, max_iterations=max_iterations, 
                         esn_burn_in=burn_in, random_seed=seed, verbose=verbose, 
                         log_space=log_space, **kwargs)
        
        # Save out weights for later
        self.readouts = readouts
        self.responsibilities = responsibilities
        
        # Set objective accordingly
        if self.readouts is None:
            self.objective_sampler = self.k_folds_objective
        else:
            self.objective_sampler = self.clustering_objective

    def clustering_objective(self, parameters):
        # Get arguments
        arguments = self.construct_arguments(parameters)
        
        # Make simple sycle reservoir
        scr = SimpleCycleReservoir(**arguments)
        
        # How many series doe we have
        n_series = self.x.shape[1]
        k_clusters = self.readouts.shape[1]
        
        # Simple check
        assert(n_series == self.y.shape[1])
        
        # Placeholder
        scores = np.zeros((n_series, k_clusters), dtype=float)
        
        # Generate error for every series
        for n in range(n_series):
            # Get series i
            x = self.x[:, n].reshape(-1, 1)
            y = self.y[:, n].reshape(-1, 1)
            
            # Compute score per cluster
            for k in range(k_clusters):
                scores[n, k] = scr.test(y, x, out_weights=self.readouts[:, k], scoring_method='L2', burn_in=self.esn_burn_in)
        
        # Compute final scores
        final_score = np.sum(self.responsibilities * scores)
        
        # Inform user
        if self.verbose:
            print('Score:', final_score)
            
        return final_score.reshape(-1, 1)
        
    def k_folds_objective(self, parameters):
        """Does k-folds on the states for a set of parameters
        
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
        # Get arguments
        arguments = self.construct_arguments(parameters)

        # Build network
        esn = self.model(**arguments)
        
        # Get all series
        y_all = self.y[self.esn_burn_in:]
        
        # Syntactic sugar
        n_samples = y_all.shape[0]
        n_series = y_all.shape[1]
        fold_size = n_samples // self.cv_samples
        
        # Score placeholder
        scores = np.zeros((self.cv_samples, n_series))
        
        for n in range(n_series):
            
            # Get state for series n
            state = esn.generate_states(self, self.x[:, n], burn_in=self.esn_burn_in)
            
            for k in range(self.cv_samples):
                # Get y_n
                y = y_all[:, n]
                
                # Validation folds
                start_index = k * fold_size
                stop_index = start_index + fold_size
                
                # Indices
                validation_indices = np.arange(start_index, stop_index)
                
                # Train mask
                train_mask = np.ones(n_samples, dtype=bool)
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
                    out_weights = scipy.linalg.pinvh(ridge_x, ridge_y, rcond=1e6*np.finfo('d').eps).reshape(-1, 1)  # Robust solution if ridge_x is singular
                
                # Validation set
                validation_x = state[validation_indices]
                validation_y = y[validation_indices]
                
                # Predict
                prediction = validation_x @ out_weights
                
                # Score
                scores[k, n] = esn.error(prediction, validation_y, scoring_method=self.scoring_method)
            
        # Pass back as a column vector (as required by GPyOpt)
        mean_score = scores.mean()
        
        # Inform user
        if self.verbose:
            print('Score:', mean_score)
            
        # Return scores
        return mean_score.reshape(-1, 1)
