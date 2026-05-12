This project studies neural network architectures designed to overcome the spectral bias of standard multilayer perceptrons (MLPs), which tend to poorly approximate high-frequency components of low-dimensional signals. Recent work shows that passing inputs through a Fourier feature mapping transforms the effective neural tangent kernel (NTK) of an MLP into a stationary kernel with tunable bandwidth, significantly improving its ability to learn high-frequency functions.

In parallel, sinusoidal representation networks (SIRENs), based on periodic activation functions, provide an alternative approach for representing complex signals and their derivatives, making them particularly suitable for implicit neural representations and PDE-based problems.

Possible directions include:

- Comparing standard MLPs, Fourier feature MLPs, and SIRENs on low-dimensional regression tasks
- Empirically studying spectral bias and frequency learning dynamics
- Investigating the role of initialization and bandwidth selection
- Solving simple boundary value problems (e.g., Poisson or Helmholtz equations) using SIRENs within PINN or Deep Ritz architectures