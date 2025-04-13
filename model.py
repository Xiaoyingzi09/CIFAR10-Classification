import numpy as np

class Activation:
    """Base class for activation functions"""
    def __init__(self, func, derivative):
        self.func = func
        self.derivative = derivative

# Create activation instances instead of using dict
ReLU = Activation(
    func=lambda x: np.maximum(0, x),
    derivative=lambda x: (x > 0).astype(float)
)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

Sigmoid = Activation(
    func=lambda x: 1 / (1 + np.exp(-x)),
    derivative=lambda x: sigmoid(x) * (1 - sigmoid(x))
)

Tanh = Activation(
    func=lambda x: np.tanh(x),
    derivative=lambda x: 1 - np.tanh(x)**2
)

LeakyReLU = Activation(
    func=lambda x: np.where(x > 0, x, 0.01 * x),
    derivative=lambda x: np.where(x > 0, 1, 0.01)
)

activation_factory = {
    'relu': ReLU,
    'sigmoid': Sigmoid,
    'tanh': Tanh,
    'leaky_relu': LeakyReLU,
}

class NeuralNetwork:
    def __init__(self, layer_dims, activation=ReLU):
        self.layer_dims = layer_dims
        self.activation = activation
        self.params = self._initialize_parameters()
        self.cache = {}

    def _initialize_parameters(self):
        """Initialize network parameters"""
        np.random.seed(42)
        params = {}
        for l in range(1, len(self.layer_dims)):
            params[f'W{l}'] = np.random.randn(self.layer_dims[l-1], self.layer_dims[l]) * 0.01
            params[f'b{l}'] = np.zeros((1, self.layer_dims[l]))
        return params

    def forward(self, X):
        """Forward propagation with state tracking"""
        self.cache = {'A0': X}
        A = X
        L = len(self.params) // 2
        
        for l in range(1, L):
            Z = np.dot(A, self.params[f'W{l}']) + self.params[f'b{l}']
            A = self.activation.func(Z)
            self.cache.update({f'Z{l}': Z, f'A{l}': A})

        # Final layer with softmax
        ZL = np.dot(A, self.params[f'W{L}']) + self.params[f'b{L}']
        AL = self._softmax(ZL)
        self.cache.update({f'Z{L}': ZL, f'A{L}': AL})
        return AL

    def backward(self, y_true, reg_lambda):
        """Backward propagation with gradient calculation"""
        grads = {}
        L = len(self.params) // 2
        m = y_true.shape[0]
        
        # Softmax gradient for final layer
        dZL = self.cache[f'A{L}'] - y_true
        grads[f'dW{L}'] = np.dot(self.cache[f'A{L-1}'].T, dZL)/m + reg_lambda*self.params[f'W{L}']
        grads[f'db{L}'] = np.sum(dZL, axis=0, keepdims=True)/m
        
        # Hidden layers gradients
        for l in range(L-1, 0, -1):
            dA = np.dot(dZL, self.params[f'W{l+1}'].T)
            dZ = dA * self.activation.derivative(self.cache[f'Z{l}'])
            grads[f'dW{l}'] = np.dot(self.cache[f'A{l-1}'].T, dZ)/m + reg_lambda*self.params[f'W{l}']
            grads[f'db{l}'] = np.sum(dZ, axis=0, keepdims=True)/m
            dZL = dZ
            
        return grads

    def update_parameters(self, grads, learning_rate):
        """Update network parameters"""
        for key in self.params:
            self.params[key] -= learning_rate * grads['d' + key]
        return self

    def _softmax(self, x):
        """Softmax implementation"""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

