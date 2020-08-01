Reservoir
=========
A Python 3 toolset for creating and optimizing Echo State Networks.

>Author: Jacob Reinier Maat, Nikos Gianniotis  
>License: MIT  
>2016-2019  

Contains:
- Vanilla ESN and Simple Cyclic Reservoir architectures.
- Bayesian Optimization with optimized routines for Echo State Nets through `GPy`.
- Clustering routines to cluister time series by optimized model.

**Please cite as:**  
J. R. Maat, N. Gianniotis, “Reservoir: a Python Package to Train and Optimize Echo State Networks ,” 2017. [Online]. Available: http://github.com/https://github.com/1Reinier/Reservoir

The open source code in this package supplements:  
J. R. Maat, N. Gianniotis and P. Protopapas, "Efficient Optimization of Echo State Networks for Time Series Datasets," 2018 International Joint Conference on Neural Networks (IJCNN), Rio de Janeiro, 2018, pp. 1-7.

## Example Use
```python
# Load data
data = np.loadtxt('example_data/MackeyGlass_t17.txt')
train = data[:4000].reshape(-1, 1)
test = data[4000:4100].reshape(-1, 1)

# Set optimization bounds
bounds = {'input_scaling': (0, 1),
          'feedback_scaling': (0, 1),
          'leaking_rate': (0, 1),
          'spectral_radius': (0, 1.25),
          'regularization': (-12, 1),
          'connectivity': (-3, 0),
          'n_nodes': (100, 1500)}

# Set optimization parameters
esn_cv = EchoStateNetworkCV(bounds=bounds,
                            initial_samples=100,
                            subsequence_length=1000,
                            eps=1e-3,
                            cv_samples=1,
                            max_iterations=1000,
                            scoring_method='tanh',
                            verbose=True)

# Optimize
best_arguments = esn_cv.optimize(y=train)

# Build best model
esn = EchoStateNetwork(**best_arguments)
esn.train(y=train)
score = esn.test(y=test, scoring_method='nrmse')

# Diagnostic plot
plt.plot(esn.predict(100), label='Predicted')
plt.plot(test, label='Ground truth')
plt.title('Prediction on next 100 steps')
plt.legend()
plt.show()

# Print the score
print("Test score found:", score)

```

