import numpy as np
import GPyOpt
import GPflow
import copy


__all__ = ['GPflowModel'] # TODO: MCMC


class GPflowModel(GPyOpt.models.GPModel):
    """GPModel class tailored towards Echo State Network optimization.
    
    Parameters
    ----------
    model_class : GPflow model class
        Which Gaussian Process model to use
    normalize_Y : bool (default True)
        Normalization on or off
    **kwargs : dict
        Extra keywords to pass to the model class in instantiation
    
    """
    analytical_gradient_prediction = True
    
    def __init__(self, model_class=GPflow.gpr.GPR, normalize_Y=True, **kwargs):
        self.normalize_Y = normalize_Y
        self.model_class = model_class
        self.model = None
        self.kwargs = kwargs
        
    def _build_model(self, X, Y):
        # Initialize model
        self.input_dim = X.shape[1]
        kernel = GPflow.kernels.Matern52(input_dim=self.input_dim, ARD=True, name="Matern52")
        self.model = model_class(x, y, kern=kernel, name='Gaussian Process', **self.kwargs)
        
        # Set proper prior close to Jeffrey's prior (1 / sigma)
        prior = lambda: GPflow.priors.Gamma(1e-3, 1e3)  # parametrization: k, theta
        self.model.kern.variance.prior = prior()
        self.model.kern.lengthscales.prior = prior()
        self.model.likelihood.variance.prior = prior()
    
    def updateModel(self, X_all, Y_all, X_new, Y_new):
        "Augment the dataset of the model"
        # Normalize if needed
        if self.normalize_Y:
            mean = Y_all.mean()
            std = Y_all.std(ddof=1)
            Y_all -= mean
            Y_all /= std
        
        # Set model
        if self.model is None:
            self._build_model(X_all, Y_all)
        else:
            self.model.X = X_all
            self.model.Y = Y_all
        
        # Optimize
        self._last_optimization_result = self.model.optimize(method='SLSQP', 
                                                             maxiter=1000, 
                                                             tol=.01)
    def predict(self, X):
        "Get the predicted mean and std at X."
        # Expand if needed
        if X.ndim == 1:
            X = X[np.newaxis,:]
        
        # Predict
        mean, variance = self.model.predict_f(X)
        std = np.sqrt(np.clip(v, 1e-20, np.inf))
        return mean, std

    def predict_withGradients(self, X):
        "Get the gradients of the predicted mean and variance at X."
        # Expand if needed
        if X.ndim == 1:
            X = X[np.newaxis,:]
        
        # Predict
        mean, variance = self.model.predict(X)
        std = np.sqrt(np.clip(v, 1e-20, np.inf))
        
        # Get gradients
        dm_dx, dv_dx = self.model.predictive_gradients(X)
        dm_dx = dm_dx[:,:,0]
        ds_dx = dv_dx / (2 * std)
        return mean, std, dm_dx, ds_dx
    
    def get_fmin(self):
        "Get the minimum of the current model."
        means, _ = self.model.predict(self.model.X)
        return means.min()
        
    def copy(self):
        "Makes a safe copy of the model."
        return copy.deepcopy(self)
        
    def get_model_parameters(self):
        "Returns a 2D numpy array with the parameters of the model"
        return np.atleast_2d(self.model.get_free_state())

    def get_model_parameters_names(self):
        "Returns a list with the names of the parameters of the model"
        return list(self.model.get_parameter_dict().keys())
