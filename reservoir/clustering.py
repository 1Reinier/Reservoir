import numpy as np
from .esn_cv import *
from .scr import *


__all__ = ['ClusteringBO']


class ClusteringBO(EchoStateNetworkCV):
    
    def __init__(self, bounds, eps=1e-3, initial_samples=100, max_iterations=300, log_space=True,
                 burn_in=30, seed=123, verbose=True, **kwargs):
        
        super().__init__(bounds, subsequence_length=-1, model=SimpleCycleReservoir, eps=eps, 
                         initial_samples=initial_samples, max_iterations=max_iterations, 
                         esn_burn_in=burn_in, random_seed=seed, verbose=verbose, 
                         esn_feedback=False, log_space=log_space, **kwargs)

    def objective_sampler(self, parameters):
        print('Works!!!')
        return np.random.uniform(size=(1, 1))
