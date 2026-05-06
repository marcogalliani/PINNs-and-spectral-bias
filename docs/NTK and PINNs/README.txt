This project investigates the training dynamics of Physics-Informed Neural Networks (PINNs) through the lens of the Neural Tangent Kernel (NTK). Although PINNs have shown strong empirical performance for solving forward and inverse PDE problems, they are often difficult to train and may fail to converge in practice. Understanding these pathologies is crucial for their reliable use in scientific machine learning.

Using NTK theory, one can analyze PINNs in the infinite-width limit and study how different components of the loss function (e.g., PDE residual, boundary conditions, data mismatch) converge at different rates. This perspective reveals an imbalance in the training dynamics that may lead to slow convergence or failure. The project may also explore adaptive training strategies based on NTK spectral properties.

Possible directions include:

- Deriving and studying the NTK for a simple PINN architecture
- Empirically analyzing the convergence rates of different loss terms
- Implementing adaptive loss-weighting strategies inspired by NTK eigenvalues
- Comparing standard PINN training with NTK-informed optimization methods

NB: The focus of the project is on the paper Wang et al. The paper Jacot et al. serves as a support for understanding.