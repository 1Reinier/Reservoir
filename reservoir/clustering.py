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
    
    def __init__(self, bounds, readouts, responsibilities, eps=1e-3, initial_samples=100, max_iterations=300, log_space=True,
                 burn_in=30, seed=123, verbose=True, **kwargs):
        
        # Initialize optimizer
        super().__init__(bounds, subsequence_length=-1, model=SimpleCycleReservoir, eps=eps, 
                         initial_samples=initial_samples, max_iterations=max_iterations, 
                         esn_burn_in=burn_in, random_seed=seed, verbose=verbose, 
                         log_space=log_space, **kwargs)
        
        # Save out weights for later
        self.readouts = readouts
        self.responsibilities = responsibilities

    def objective_sampler(self, parameters):
        # Make simple sycle reservoir
        scr = SimpleCycleReservoir(**parameters)
        
        # How many series doe we have
        n_series = self.x.shape[1]
        k_clusters = self.readouts.shape[1]
        
        # Simple check
        assert(n_series == self.y.shape[1])
        
        # Placeholder
        scores = np.zeros((n_series, k_clusters), dtype=float)
        
        # Generate error for every series
        for n in n_series:
            # Get series i
            x = self.x[:, n].reshape(-1, 1)
            y = self.y[:, n].reshape(-1, 1)
            
            # Compute score per cluster
            for k in k_clusters:
                scores[n, k] = scr.test(y, x, out_weights=self.readouts[:, k], scoring_method='L2', burn_in=self.esn_burn_in)
        
        # Compute final scores
        final_score = np.sum(self.responsibilities * scores)
        
        # Inform user
        if self.verbose:
            print('Score:', final_score)
            
        return final_score
