# From-Scratch Multi-Layer Perceptron (MLP)

A Python implementation of a Multi-Layer Perceptron (MLP) built from scratch using NumPy, designed to solve non-linearly separable problems like the XOR gate.

- **Pure NumPy Implementation:** Forward pass and backpropagation built natively without external deep learning frameworks.
- **Hyperparameter Tuning:** Custom learning rate grid search to automatically evaluate and select the best model hyperparameters based on loss.

## Repository Structure

```text
From-Scratch-MLP/
├── src/
│   ├── activations.py     # Activation functions (Sigmoid, derivative)
│   ├── loss.py            # Loss functions (MSE, derivative)
│   ├── mlp.py             # Core MLP architecture & backpropagation
│   └── tuning.py          # Hyperparameter tuning algorithm
│
├── xor_mlp_scratch.py     # Standalone single-file demonstration
├── requirements.txt
└── README.md
```

## Results (XOR Problem)

| Input 1 | Input 2 | Target | Prediction |
|:-------:|:-------:|:------:|:----------:|
| 0 | 0 | 0 | ~0.0008 | 
| 0 | 1 | 1 | ~0.9980 | 
| 1 | 0 | 1 | ~0.9980 | 
| 1 | 1 | 0 | ~0.0031 | 
