#import pytest
import numpy as np
import matplotlib.pyplot as plt
from reservoir.esn import *


def test_esn():
    # Load data
    data = np.loadtxt('example_data/MackeyGlass_t17.txt')
    train = data[:4000].reshape(-1, 1)
    test = data[4000:4500].reshape(-1, 1)
    
    # Set optimization bounds
    bounds = [{'name': 'input_scaling', 'type': 'continuous', 'domain': (0, 1)},
              {'name': 'feedback_scaling', 'type': 'continuous', 'domain': (0, 1)},
              {'name': 'leaking_rate', 'type': 'continuous', 'domain': (0, 1)}, 
              {'name': 'spectral_radius', 'type': 'continuous', 'domain': (0, 1.25)},
              {'name': 'regularization', 'type': 'continuous', 'domain': (-12, 1)},
              {'name': 'connectivity', 'type': 'continuous', 'domain': (-3, 0)},
              {'name': 'n_nodes', 'type': 'continuous', 'domain': (100, 1500)}]
    
    # Set optimization parameters
    esn_cv = EchoStateNetworkCV(bounds=bounds,
                                initial_samples=100,
                                subsequence_length=1000,
                                eps=1e-3,
                                #mcmc_samples=20,  # Enable to test MCMC sampling (memory usage high!)
                                cv_samples=1, 
                                max_iterations=1000, 
                                scoring_method='tanh',
                                verbose=True
                                )
    
    # Optimize
    best_arguments = esn_cv.optimize(y=train)
    
    # Build best model
    esn = EchoStateNetwork(**best_arguments)
    esn.train(y=train)
    score = esn.test(y=test, scoring_method='nrmse')
    
    # Diagnostic plot
    plt.plot(esn.predict(100), label='Predicted')
    plt.plot(test[:100], label='Ground truth')
    plt.title('Prediction on next 100 steps')
    plt.legend()
    plt.show()
    
    # Final test
    print("Test score found:", score)
    assert score < 1e-2, 'ESN-CV does not meet required score'
    
