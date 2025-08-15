import torch
from torch import optim
from rk4 import StateSpaceModel

def generate_spring_data(device, t0, tn, dt, y0, forcing_function, save=True):
    # Here we define a model with known dynamics to generate our "truth data".

    # Assume mass of 1 kg is displaced by 1 meter with no input except gravity
    m = 1  # spring mass
    c = 1  # damping coefficient
    k = 9.8  # spring constant

    # State matrices of linearly damped spring
    A = torch.tensor([[0, 1], [-k / m, -c / m]])  # system matrix (n_states x n_states)
    B = torch.tensor([[0], [1 / m]])  # input matrix  (m_inputs x 1)
    C = torch.tensor([[1.0, 0], [0, 1]])  # pass through both states (p_outputs x n_states)
    D = torch.zeros((2, 1))  # input does not bypass dynamics (p_outputs x m_inputs)

    # instantiate SS model with accurate matrices
    optimizer = optim.Adam
    true_model = StateSpaceModel(A, B, C, D, forcing_function=forcing_function, opt=optimizer).to(device)

    # run the underlying integrator to generate solution data to use in training.
    solution = true_model.run(device, t0, tn, dt, y0)

    if save:
        torch.save(solution, "solution.pt")
        torch.save(A, "A.pt")
        torch.save(B, "B.pt")
        torch.save(C, "C.pt")
        torch.save(D, "D.pt")

    return solution, A, B, C, D
