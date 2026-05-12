# PINNs-and-spectral-bias
This repo aims at investigating the spectral bias phenomenon in neural network training with a particular focus on how the issue translates to Physics-Informed NN architectures. 

Moreover, the repo aims at comparing PINNs to physics-informed statistical methods.

## Code structure
- `src`: general utilities
- `examples`: python or R scripts to showcase intersiting phenomena
- `docs`: papers and documentation

## Submodules
- `ode-fit`: my repo implemening ODE smoothing
- `SpectralBias`: (Rahaman, 2019) experiments on spectral bias

## Project evaluation
- exam date: 4/6 or 24/6
- to partecipate in the exam: compile [this form](https://forms.office.com/pages/responsepage.aspx?id=K3EXCvNtXUKAjjCd8ope6xofTnlQ7dJGoQ_D1d82PblURU5DSkcxV1g2UkNBUFAwUFpQNVdOWUJTMC4u&route=shorturl)
- compile the [excel](https://polimi365-my.sharepoint.com/:x:/g/personal/10377072_polimi_it/IQD2GT5XbdOOTISRbIGbkPxwAW_hWJJGU7GOY6jibJw6p80?e=0AmdPJ) with project title 

## Ideas
- The eigenvalues of the Neural Tangent Kernel seems to govern how the PINN learn a function, namely which frequencies are learned first. It seems that the Optimal Control Approach to solve the regularized problem presents the same learning behaviour (in fact, [Li et al, 2018]() shows that the NNs learning problem can be formulated as Optimal Control Problem). The idea is to see how the same concept of NTK applies to the optimal control approach.
- Implement revised approaches to mitigate the spectral bias
- Approaches:
  - fourier features
  - siren
  - neural ODE
  - 