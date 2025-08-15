from functools import partial
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

from typing import Type

"""
This code was written as part of a larger project to develope a model that accurately describes the biological 
interractions between a pest insect and its habitat, namely the eastern spruce budworm and the spruce forests it 
inhabits. The model is developed in the written paper, and here the system of equations is solved and displayed 
graphically. The solution is calculated and displayed using the below define Runge-Kutta 4 numerical method and 
matplotlib library.
"""


def rk4_step(t_i, y_i, dt_i, field):
    k1 = dt_i * field(t_i, y_i)
    k2 = dt_i * field(t_i + 0.5 * dt_i, y_i + 0.5 * k1)
    k3 = dt_i * field(t_i + 0.5 * dt_i, y_i + 0.5 * k2)
    k4 = dt_i * field(t_i + dt_i, y_i + k3)
    y_next = y_i + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return y_next


def run_rk4(field, t_0, t_n, dt, y_0, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    :param field: method - the vector field y' = f(t, y)
    :param t_0: float - start time
    :param t_n: float - stop time
    :param dt: float - the timestep
    :param y_0: array - contains initial conditions
    :return: generator
    """

    # Initialize solution matrix. Each row is the solution to the system
    # for a given time step. Each column is the full solution for a single
    # equation.
    y_i = y_0

    time = torch.arange(t_0, t_n + dt, dt).to(device)

    for i, t_i in enumerate(time):
        y_i = rk4_step(t_i, y_i, dt, field)
        t_i += dt
        yield y_i


def ode(system_matrix, input_matrix, u, time, state):
    return system_matrix @ state + input_matrix @ u(time)


class StateSpaceModel(nn.Module):
    """
    Args:
        system_matrix (Tensor): (n_states x n_states) Governs how the states evolve over time
        input_matrix (Tensor): (n_states x m_inputs) Maps affect of external inputs to changes in state
        output_matrix (Tensor): (m_inputs x p_outputs) Maps internal states to external outputs
        feedthru_matrix (Tensor): (p_outputs x m_inputs) Maps affect of external forces to changes in outputs
    """

    def __init__(self,
                 system_matrix,
                 input_matrix,
                 output_matrix,
                 feedthru_matrix,
                 forcing_function,
                 opt: Type[optim.Optimizer] = optim.Adam):

        super().__init__()

        # Register State Space Matrices as learnable parameters.
        self.A = nn.Parameter(system_matrix)
        self.B = nn.Parameter(input_matrix)
        self.C = nn.Parameter(output_matrix)
        self.D = nn.Parameter(feedthru_matrix)

        # register them to named_parameters as well.
        self.register_parameter("System Matrix", self.A)
        self.register_parameter("Input Matrix", self.B)
        self.register_parameter("Output Matrix", self.C)
        self.register_parameter("Feedthrough Matrix", self.D)
        self.u = forcing_function

        # Register the parameters with the optimizer
        self.opt = opt
        self.integrand = partial(ode, system_matrix, input_matrix, forcing_function)

    def forward(self, t_i: torch.Tensor, x_i: torch.Tensor, dt):
        """

        Args:
            t_i: The current time
            x_i: The current state
            dt: timestep

        Returns:
            state: The new state.
            out: the new state space output.

        """
        integrand = partial(ode, self.A, self.B, self.u)
        state = rk4_step(t_i, x_i, dt, integrand)
        out = self.C @ state + self.D @ self.u(t_i)

        return state, out

    def run(self, device, t_0, t_n, dt, y_0):
        """
        Run the solver continuously for a set amount of time.

        Args:
            device: CUDA or CPU
            t_0: start time
            t_n: stop time
            dt: time step size
            y_0: initial conditions

        Returns:
            State space system's outputs
        """

        with torch.no_grad():
            # move parameters to appropriate device memory
            y_0 = y_0.to(device)
            time = torch.arange(t_0, t_n + 2 * dt, dt).to(device)

            # initialize the solution tensor from initial state
            rows = y_0.dim()
            columns = int(2 + (t_n - t_0) // dt)
            solution = torch.ones((rows, columns), device=device) * (self.C @ y_0 + self.D @ self.u(time[0]))

            # run integrator
            integrand = partial(ode, self.A, self.B, self.u)
            for i, state in enumerate(run_rk4(integrand, t_0, t_n, dt, y_0), start=1):
                # derive outputs from integrated state, add to solution tensor
                out = self.C @ state + self.D @ self.u(time[i])
                solution[:, i:i + 1] = out

        return solution


    def optimize(self, device, y_init, dt, data_loader, epocs, lr=0.001):
        """
        Optimizes state space matrices via back propagation.

        Args:
            device: Device to train parameters on.
            y_init: Initial state of the system.
            dt: The timestep of the underlying solver. Should match the timestep in the dataset.
            data_loader: Contains real world data (t, y(t))
            epocs: The number of times the entire dataset is iterated over during training.
            lr: learning rate
        """
        torch.autograd.set_detect_anomaly(True)
        y_init = y_init.to(device)

        opt = self.opt(self.parameters(), lr=lr)
        loss = []

        for i in range(epocs):
            epoch_loss = self.train_one_epoch(device, y_init, dt, data_loader, opt)
            loss.append(epoch_loss)
            print(f"Epoch {i} loss: {epoch_loss}")

        return loss


    def train_one_epoch(self, device, y_init, dt, data_loader, opt):

        curr_states = y_init  # TODO: add check to make sure input initial state equals target initial state

        running_loss = 0.0
        for i, (t_i, target_state) in enumerate(data_loader):
            t_i, target_state = t_i.to(device), target_state.to(device)
            opt.zero_grad()

            y_states, y_output = self.forward(t_i, curr_states, dt)
            loss = nn.functional.mse_loss(y_output, target_state.squeeze(0))
            loss.backward()
            opt.step()

            curr_states = y_states.detach()
            running_loss += loss.item()

        return running_loss / len(data_loader)


if __name__ == '__main__':

    from utils.datagen import generate_spring_data
    from utils.datasets import TensorData
    import matplotlib.pyplot as plt
    from pathlib import Path

    device_type = "cuda" if torch.cuda.is_available() else "cpu"

    # No external forces
    def forcing_func(x):
        return torch.tensor([[0.0]]).to(device_type)

    # rk4 run parameters
    t0 = 0  # start time
    tn = 10  # stop time
    delta_t = 0.1  # timestep

    # initial conditions
    y0 = torch.tensor([[2.0], [0.0]])  # position and velocity

    # if we have not already generated "real world" truth data, do so.
    if Path("solution.pt").exists():
        true_solution = torch.load("solution.pt")
        A = torch.load("A.pt")
        B = torch.load("B.pt")
        C = torch.load("C.pt")
        D = torch.load("D.pt")
    else:
        # generate data and the associated state matrices
        true_solution, A, B, C, D = generate_spring_data(device_type, t0, tn, delta_t, y0, forcing_func, save=True)

    # plot solution
    true_solution = true_solution.detach().to('cpu')

    error = partial(torch.normal, 0.0, 0.33)
    A = A + error(A.shape)
    B = B + error(B.shape)
    C = C + error(C.shape)
    D = D + error(D.shape)

    t = torch.arange(t0, tn + delta_t, delta_t)

    # run unoptimized model to get estimated solution.
    est_model = StateSpaceModel(A, B, C, D, forcing_func).to(device_type)
    est_solution = est_model.run(device_type, t0, tn, delta_t, y0).detach().to('cpu')

    # load training data into data loader
    truth_data = torch.vstack([t, true_solution])
    dataset = TensorData(truth_data, 1, arrangement="long")

    # optimize the model based on new "real world" data
    training_loader = DataLoader(dataset, batch_size=1)
    est_model.optimize(device_type, y0, delta_t, training_loader, lr=0.003, epocs=2000)

    # get estimated solution from updated model
    new_est_solution = est_model.run(device_type, t0, tn, delta_t, y0).detach().to('cpu')

    # plot the true and estimated solutions side by side.
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4)

    # plot pre-optimization results next to truth data
    ax1.plot(t, true_solution[0], label="position", c="blue")
    ax2.plot(t, true_solution[1], label="velocity", c="orange")
    ax1.plot(t, est_solution[0], "--", c="cornflowerblue", label="old est position")
    ax2.plot(t, est_solution[1], "--", c="wheat", label="old est velocity")

    # plot the post-optimized data to see the improvement next to real-world data.
    ax3.plot(t, true_solution[0], label="position", c="blue")
    ax4.plot(t, true_solution[1], label="velocity", c="orange")
    ax3.plot(t, new_est_solution[0], "--", c="cornflowerblue", label="new est position")
    ax4.plot(t, new_est_solution[1], "--", c="wheat", label="new est velocity")

    loc = "lower right"
    ax1.legend(loc=loc)
    ax2.legend(loc=loc)
    ax3.legend(loc=loc)
    ax4.legend(loc=loc)
    plt.tight_layout()
    plt.show()

    true_params = (A, B, C, D)
    for p1, p2 in zip(est_model.named_parameters(), true_params):
        print(f" Comparison of Matrix {p1[0]}:")
        print(p1[1])
        print(p2)
        print("\n")
