This project explores the connection between deep learning, optimal control, and partial differential equations. Neural network training can be formulated as a discrete-time optimal control problem, where the network layers define the system dynamics and the loss function plays the role of a terminal cost. This viewpoint leads to necessary optimality conditions via Pontryagin's Maximum Principle and motivates alternative training algorithms, such as the method of successive approximations (MSA), which do not rely directly on standard backpropagation gradients.

In parallel, recent work interprets optimization methods such as Entropy-SGD as gradient descent on a modified loss function characterized by a viscous Hamilton–Jacobi PDE. This PDE and stochastic control perspective provides insight into the geometry of the loss landscape and the convergence properties of training algorithms.

Possible directions include:

- Formulating a simple neural network training problem as an optimal control problem
- Implementing and testing a control-inspired training scheme (e.g., discrete MSA)
- Studying relaxed loss landscapes arising from Hamilton–Jacobi PDEs
- Exploring discrete-weight or sparse neural networks from an optimal control viewpoint