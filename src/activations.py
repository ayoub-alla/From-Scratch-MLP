import numpy as np

#____ activation function sigmoid
def sigmoid(x: np.ndarray) -> np.ndarray:
    
    # verfication de l entree 
    assert isinstance(x, np.ndarray), "error - L'entrée doit être tableau numpy"
    
    # function
    result = 1 / (1 + np.exp(-x))
    
    # verification de sortie
    assert np.all((result >= 0) & (result <= 1)), "error - résultat devrait situer entre 0 et 1"
                  
    return result

#____ derivee function sigmoid
def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
    
    # verfication de l entree 
    assert isinstance(x, np.ndarray), "error - L'entrée doit être tableau numpy"
    
    # function
    result = sigmoid(x) * (1 - sigmoid(x))
    
    # verification de sortie
    assert np.all(result >= 0), "error - résultat devrait plus ou egale a 0"

    return result

# fonction d'activation ReLU 
def relu(x: np.ndarray) -> np.ndarray:
    
    assert isinstance(x, np.ndarray), "error - L'entrée doit être tableau numpy"
    return np.maximum(0, x)

# fonction d'activation Softmax
def softmax(x: np.ndarray) -> np.ndarray:
    
    assert isinstance(x, np.ndarray), "error - L'entrée doit être tableau numpy"
    assert x.ndim >= 1, "error - L'entrée doit avoir au moins plus d'un dimension"
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    result = exp_x / exp_x.sum(axis=-1, keepdims=True)
    assert np.allclose(result.sum(axis=-1), 1.0), "error - somme du softmax doit être égale à 1 sur le dernier axe"
    return result

# fonction d'activation Leaky relu
def leaky_relu(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:

    assert isinstance(x, np.ndarray), "error - L'entrée doit être  tableau numpy"
    assert alpha > 0, "error - alpha doit être  > 0"
    return np.where(x > 0, x, alpha * x)