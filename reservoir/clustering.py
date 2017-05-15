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
    
    def __init__(self, bounds, responsibilities, readouts=None, eps=1e-3, initial_samples=100, max_iterations=300, log_space=True,
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
                if self.readouts is None:
                    # Validation set
                    cutoff = int((1 - self.validate_fraction) * x.shape[0])
                    if cutoff >= self.esn_burn_in:
                        raise ValueError("Validation set shorter than ESN burn in")
                    
                    # Train model
                    scr.train(y[:cutoff], x[:cutoff], burn_in=self.esn_burn_in)
                    
                    # Validation score
                    scores[n, k] = scr.test(y[cutoff:], x[cutoff:], scoring_method='L2', burn_in=self.esn_burn_in)
                else:
                    scores[n, k] = scr.test(y, x, out_weights=self.readouts[:, k], scoring_method='nmse', burn_in=self.esn_burn_in)
        
        # Checks
        assert(np.all(np.isfinite(scores)))
        
        # Compute final scores
        final_score = np.sum(self.responsibilities * scores)
        
        # Inform user
        if self.verbose:
            print('Score:', final_score)
            
        return final_score
