from .esn import EchoStateNetwork
import numpy as np


__all__ = ['SimpleCycleReservoir']


class SimpleCycleReservoir(EchoStateNetwork):
    
    def __init__(self, n_nodes=30, regularization=1e-8, cyclic_weight=0.5, input_weight=0.5, seed=42):
        # Initialize ESN
        super().__init__(n_nodes=n_nodes, 
                        random_seed=seed,
                        regularization=regularization, 
                        leaking_rate=1., 
                        input_scaling=None,
                        feedback_scaling=None, 
                        spectral_radius=None, 
                        connectivity=None,
                        feedback=False)
        
        # Save attributes
        self.n_nodes = n_nodes
        self.regularization = regularization
        self.cyclic_weight = cyclic_weight
        self.input_weight = input_weight
        self.seed = seed
        
        # Generate reservoir
        self.generate_scr_reservoir()
        
    def generate_scr_reservoir(self):
        random_state = np.random.RandomState(self.seed)
        
        # Set reservoir weights
        self.weights = np.zeros((self.n_nodes, self.n_nodes))
        for i in range(self.n_nodes - 1):
            self.weights[i+1, i] = self.cyclic_weight
        self.weights[0, -1] = self.cyclic_weight
        
        # Default state
        self.state = np.zeros((1, self.n_nodes))
        
        # Set out to none to indicate untrained ESN
        self.out_weights = None
        
    def train(self, y, x=None, burn_in=30):
        return super().train(y=y, x=x, burn_in=100, input_weight=self.input_weight)
