import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

"""
PINN methods were taken from a simple model for an ODE in 'A Hands-on Introduction to Physics-Informed 
Neural Networks' on NanoHub by Atharva Hans and Ilias Bilionis, and applied to Burgers' equation and validated
with the finite difference method.

We will be solving the 1D Burgers' equation:

∂Ψ/∂t + Ψ⋅∂Ψ/∂x = ν⋅∂²Ψ/∂x²,   x ∈ [0,L], t ∈ [0,T]
Ψ(0, t) = A,
Ψ(L, t) = B,
Ψ(x, 0) = f(x) = 3⋅exp(-30⋅(x-0.2)²),

We will write the solution as the trial function

Ψ̂(x, t; θ) = A⋅(L - x) + B⋅x + N(x, t; θ)⋅(L - x)⋅x,   

for a neural network, N(x, t; θ). The solution must satisfy ∂Ψ/∂t + Ψ⋅∂Ψ/∂x - ν⋅∂²Ψ/∂x² = 0 and Ψ(x, 0) - f(x) = 0, so 
we will define the loss as 

L_1(θ) = ∫_0^L ∫_0^T [∂Ψ̂(x, t; θ)/∂t - ∂Ψ̂(x, t; θ)/∂x]² dt dx,

L_2(θ) = ∫_0^L [Ψ̂(x, 0; θ) - f(x; θ)]² dx,

which the neural network will be trained to minimize.

The initial condition is a Gaussian curve and such that a shock wave nearly forms, but not quite, allowing for an interesting
solution that is still easy to solve for without shock handling.

The model is not great when compared to the FDM validator, but the overall shape of the solution is reasonably alike to the actual 
solution. I have played with the hyperparameters, but I can't quite get a better model than what is presented here.
"""

# Hyperparameters -------------------------------------------------------------
ν = 0.0075  # diffusion coefficient 
A, B = 0, 0 # Dirichlet boundaries  Ψ(0,t)=A,  Ψ(L,t)=B
L, T = 1.0, 1.0  # spatial length & end‑time
f = lambda x: 3 * torch.exp(-30*(x-0.2)**2) # initial condition

N_x, N_t = 100, 1500  # FD grid (for validation)
N_pts = 500  # random collocation pts (training)
NEURONS = 100  # hidden‑layer width
EPOCHS = 100  # iterations

torch.manual_seed(0)
np.random.seed(0)

x = np.linspace(0.0, L, N_x, dtype=float)
t = np.linspace(0.0, T, N_t, dtype=float)
Δx = x[1] - x[0]
Δt = t[1] - t[0]
x_torch = torch.rand((N_pts, 1), requires_grad=True)
t_torch = torch.rand((N_pts, 1), requires_grad=True)

Psi_trial = lambda x, t: A * (L - x) + B * x + N(torch.cat([x, t], dim=1))[:, 0:1] * (L - x) * x

# Physics Informed Neural Network ---------------------------------------------
N = nn.Sequential(
    nn.Linear(2, NEURONS),
    nn.Tanh(),
    nn.Linear(NEURONS, NEURONS),
    nn.Sigmoid(),
    nn.Linear(NEURONS, NEURONS),
    nn.Sigmoid(),
    nn.Linear(NEURONS, 1)
)

# Loss & Optimizer ------------------------------------------------------------
def residual_loss(x, t):  # we want to satisfy ∂Ψ/∂t + Ψ⋅∂Ψ/∂x - ν⋅∂²Ψ/∂x² = 0
    x.requires_grad = True
    t.requires_grad = True
    Psi = Psi_trial(x, t)

    Psi_t = torch.autograd.grad(Psi, t, torch.ones_like(Psi), create_graph=True)[0]
    Psi_x = torch.autograd.grad(Psi, x, torch.ones_like(Psi), create_graph=True)[0]
    Psi_xx = torch.autograd.grad(Psi_x, x, torch.ones_like(Psi_x), create_graph=True)[0]
    
    return torch.mean((Psi_t + Psi * Psi_x - ν * Psi_xx) ** 2)

def initial_loss(x):  # we want to satisfy Ψ(x, 0) - f(x) = 0
    t_0 = torch.zeros_like(x)

    return torch.mean((Psi_trial(x, t_0) - f(x)) ** 2)

def total_loss(x, t):
    return residual_loss(x, t) + initial_loss(x)

optimizer1 = torch.optim.Adam(N.parameters(), lr=1e-3)  # two optimizers seems to improve the model
optimizer2 = torch.optim.LBFGS(N.parameters())

# Training Loop ---------------------------------------------------------------
for epoch in range(EPOCHS): 
    # 1. & 2. Forward Pass & Loss
    loss = total_loss(x_torch, t_torch)

    # 3. Optimizer Zero Grad
    optimizer1.zero_grad()

    # 4. Backpropagation
    loss.backward()

    # 5. Gradient Descent
    optimizer1.step()

def closure():  # must have a closure function written like this for LBFGS to work
    # 1. & 2. Forward Pass & Loss
    loss = total_loss(x_torch, t_torch)  

    # 3. Optimizer Zero Grad
    optimizer2.zero_grad()

    # 4. Backpropagation
    loss.backward()
    return loss

for epoch in range(EPOCHS):
    # 5. Gradient Descent
    optimizer2.step(closure)

# Finite Difference Validation ------------------------------------------------
Psi_fd = np.zeros((N_t, N_x), dtype=float)  # [t, x]

f = lambda x: 3 * np.exp(-30*(x-0.2)**2)
Psi_fd[0, :] = f(x)

r1 = ν * Δt / (Δx ** 2)
r2 = Δt / Δx 
for j in range(0, N_t - 1):  # time
    u = Psi_fd[j, :]

    diffusion = (u[2:] - 2*u[1:-1] + u[0:-2]) / (Δx**2)
    convection = u[1:-1] * (u[1:-1] - u[0:-2]) / Δx

    Psi_fd[j+1, 1:-1] = u[1:-1] + Δt * (ν * diffusion - convection)

    Psi_fd[j+1, 0] = A
    Psi_fd[j+1, -1] = B
    
# Plot & Error ----------------------------------------------------------------
print("")
print(f"Stability checks:")
print(f'{r1} must be less than 0.5 for stability.')
print(f'{r2} must be less than 1.0 for stability.')
print("")

x_plot = torch.linspace(0, L, N_x)[:, None]  # shape (N_x, 1)
t_plot = torch.linspace(0, T, N_t)[:, None]  # shape (N_t, 1)
x_mesh, t_mesh = torch.meshgrid(x_plot.squeeze(), t_plot.squeeze(), indexing="ij")

xt_input = torch.cat([x_mesh.reshape(-1, 1), t_mesh.reshape(-1, 1)], dim=1)

with torch.no_grad():
    Psi_pred = Psi_trial(xt_input[:, :1], xt_input[:, 1:])
    Psi_pred = Psi_pred.reshape(N_x, N_t)

# PINN
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x_mesh.numpy(), t_mesh.numpy(), Psi_pred.numpy(), cmap='viridis')
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel(r'$\Psi(x,t)$')
plt.title("PINN solution to Burgers' equation")
plt.tight_layout()
plt.show()

# FDM
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x_mesh.numpy(), t_mesh.numpy(), Psi_fd.T, cmap='coolwarm')
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel(r'$\Psi_{\text{FDM}}$')
ax.set_title("FDM solution to Burgers' equation")
plt.tight_layout()
plt.show()

# Difference
error_surface = Psi_pred.numpy() - Psi_fd.T  # Must be same shape
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x_mesh.numpy(), t_mesh.numpy(), error_surface, cmap='coolwarm')
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel(r'$\Psi_{\text{PINN}} - \Psi_{\text{FDM}}$')
ax.set_title('Difference between PINN and FDM')
plt.tight_layout()
plt.show()