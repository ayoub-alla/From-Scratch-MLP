import numpy as np
from src.mlp import MLP
from src.tuning import mlp_hyperparameter_tuning

# Donnees XOR
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# a simple tuning function for only lr
best_lr = mlp_hyperparameter_tuning(MLP , X=X, y=y, epochs=2000, learning_rates=[0.01 , 10 , 100 , 100.001  ], input_size=2, hidden_size=4, output_size=1, seed=42 )
# the best was 100 

# Creation et entrainement du MLP avec un seed
mlp = MLP(input_size=2, hidden_size=4, output_size=1, learning_rate=best_lr, seed=42)
mlp.train(X, y, epochs=10000)



mlp.save_model("xor_mlp_p.npz")

# Simple Predictions  test
predictions = mlp.forward(X)
print("\n Predictions apres entrainement :")

for i in range(len(X)):
    print(f"Entree: {X[i]}, Prediction: {predictions[i][0]:.4f}, Attendu: {y[i][0]}")
        
    # Assertion : Verifier que les predictions sont coherentes avec XOR
    if y[i][0] == 0:
        assert predictions[i][0] < 0.5, f"Prediction pour {X[i]} devrait etre < 0.5"
    else:
        assert predictions[i][0] > 0.5, f"Prediction pour {X[i]} devrait etre > 0.5"
