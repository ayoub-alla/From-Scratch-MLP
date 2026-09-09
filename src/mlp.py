import numpy as np
import json
from .activations import sigmoid , sigmoid_derivative
from .loss import MSE


#_____ MLP

class MLP:

    # constructeur de class 
    def __init__(self, input_size : int , hidden_size : int , output_size : int , 
                 learning_rate : float = 0.01 , seed : int = 42):

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # random initialisation des poids et biais
        np.random.seed(seed)
        self.W1 = np.random.randn(input_size , hidden_size)  #poid d entree
        self.b1 = np.zeros((1 , hidden_size)) # bias couche cache
        self.W2 = np.random.randn(hidden_size , output_size)   # poid de couche cache
        self.b2 = np.zeros((1 , output_size)) # bias couche sortie

        # Assertions : Verifier les dimensions
        assert self.W1.shape == (input_size, hidden_size), "Mauvaise dimension pour W1"
        assert self.b1.shape == (1, hidden_size), "Mauvaise dimensionpour b1"
        assert self.W2.shape == (hidden_size, output_size), "Mauvaise dimension pour W2"
        assert self.b2.shape == (1, output_size), "Mauvaise dimension pour b2"

    def forward(self , X):

        # Implementez la propagation avant
        self.Z1 = np.dot( X , self.W1 ) + self.b1 # Calcul de la somme ponderee ( couche cache ) 
        self.A1 = sigmoid(self.Z1) # Activation de la couche cache on applique sigmoid
        self.Z2 = np.dot( self.A1 , self.W2 ) + self.b2 # Calcul de Z2 la somme ponderee ( couche de sortie )
        self.A2 = sigmoid(self.Z2) # Activation de la couche de sortie sigmoid

        # Assertions : Verifier les dimensions
        assert self.A1.shape == (X.shape[0], self.W1.shape[1]),"Mauvaise dimension pour A1"
        assert self.A2.shape == (X.shape[0], self.W2.shape[1]), "Mauvaise dimension pour A2"

        # Assertion : Verifier que A2 est entre 0 et 1
        assert np.all((self.A2 >= 0) & (self.A2 <= 1)), "Sortie A2 doit tre entre 0 et 1"
    
        return self.A2
    
    def compute_loss(self, y_true, y_pred):
           
           # le calcul de la perte (MSE) avg de la somme de y true et y predit
           loss = MSE(y_pred=y_pred , y_true=y_true)

           # Assertion : Verifier que la perte est positive ou nulle
           assert loss >= 0, "La perte doit etre positive ou nulle"

           # Assertion : Verifier que la perte est 0 si y_true = y_pred
           if np.array_equal(y_true, y_pred):
              assert np.isclose(loss, 0), "La perte doit etre 0 si y_true = y_pred"

           return loss
    
    def backward(self, X, y, y_pred):

        m = X.shape[0]
        
        # Implementez la retropropagation 
        # Gradient de la couche de sortie 
        dZ2 = (self.A2 - y) * sigmoid_derivative(self.Z2) # Gradient de Z2
        dW2 = (1 / m) * np.dot(self.A1.T, dZ2) # Gradient de W2
        db2 = (1 / m) * np.sum(dZ2, axis=0, keepdims=True) # Gradient de b2 calcule des lines par column axis 0
        
        # Gradient de la couche cachée 
        dA1 = np.dot(dZ2, self.W2.T)  # Gradient de A1 
        dZ1 = dA1 * sigmoid_derivative(self.Z1) # Gradient de Z1
        dW1 = (1 / m) * np.dot(X.T, dZ1) # Gradient de W1
        db1 = (1 / m) * np.sum(dZ1, axis=0, keepdims=True) # Gradient de b1 
        
        # Assertions : Verifier les dimensions des gradients 
        assert dW1.shape == self.W1.shape, "Mauvaise dimension pour dW1" 
        assert db1.shape == self.b1.shape, "Mauvaise dimension pour db1" 
        assert dW2.shape == self.W2.shape, "Mauvaise dimension pour dW2" 
        assert db2.shape == self.b2.shape, "Mauvaise dimension pour db2" 
        
        # Mise a jour des poids et biais 
        self.W1 -= self.learning_rate * dW1 
        self.b1 -= self.learning_rate * db1 # A COMPLETER 
        self.W2 -= self.learning_rate * dW2 # A COMPLETER 
        self.b2 -= self.learning_rate * db2 

    def train(self, X, y, epochs): 

        print("________Training start")

        prev_loss = float('inf') 

        for epoch in range(epochs):

            y_pred = self.forward(X) # forward function
            loss = self.compute_loss(y_true = y , y_pred = y_pred) # loss functio
            
            if epoch % 500 == 0:
                print(f"Epoch {epoch} , Loss MSE : {loss}")
                
            # Assertion : Verifier que la perte diminue (ou reste stable)
            #assert loss <= prev_loss or np.isclose(loss, prev_loss), "La perte doit diminuer ou rester stable"

            
            prev_loss = loss

            # back propagation pour ajuster les poids et les bies
            self.backward(X, y, y_pred)
            
            if epoch == epochs - 1 :
                print("________Training ends")

        return loss
    

    def save_model(self, filepath: str = "model_weights.npz"):
        #model weights
        np.savez(filepath, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2)
        print(f"_______Model weights are saved in {filepath}")

