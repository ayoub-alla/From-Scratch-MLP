import numpy as np
from .mlp import MLP

# a hyperparameter tuning function for the MLP
# with just learning rate for start , this need more optimasation to be better
def mlp_hyperparameter_tuning(mlp: type[MLP] , X:np.ndarray, y:np.ndarray, epochs:int, learning_rates:list, *args, **kwargs) -> float:

    best_lr = 0.0
    best_loss = float('inf')

    print("_____Finding the best Learning Rate start :")
    # the mlp train returns the final loss
    for lr in learning_rates:

        c = mlp( learning_rate = lr , *args , **kwargs)
        loss = c.train(X, y, epochs)
        print(f"_____End of Testing with lr : {lr} , loss : {loss}")

        if loss < best_loss : 
            best_loss = loss
            best_lr = lr


    print(f"_____Best lr : {best_lr} , loss : {best_loss}")
    print("____________________End of lr tuning")

    return best_lr

        
        
      