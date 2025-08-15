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

    def forward(self, time: torch.Tensor, state: torch.Tensor):
        """
        Execute the state space equations with a given state vector. Calculates both the
        state derivative and the system output.

        Args:
            time: The current time.
            state: The current state, or solution of the system.

        Returns:
            dx: The derivative of the state
            out: the new state space output.

        """

        dx = self.A @ state + self.B @ self.u(time)
        out = self.C @ state + self.D @ self.u(time)

        return dx, out

    def run(self, device, t_0, t_n, dt, y_0, compute_output=False):
        """
        Run the solver continuously for a set amount of time. Returns

        Args:
            device: CUDA or CPU
            t_0: start time
            t_n: stop time
            dt: time step size
            y_0: initial conditions
            compute_output: Whether to transform the states with C and D matrices.

        Returns:
            State space system's outputs
        """

        with torch.no_grad():
            # move parameters to appropriate device memory
            y_0 = y_0.to(device)
            time = torch.arange(t_0, t_n + 2 * dt, dt).to(device)

            # initialize the tensor containing states
            state_rows = self.A.size(-1)
            columns = int(2 + (t_n - t_0) // dt)
            states = torch.ones((state_rows, columns), device=device) * y_0

            # run integrator
            integrand = partial(ode, self.A, self.B, self.u)
            for i, state in enumerate(run_rk4(integrand, t_0, t_n, dt, y_0), start=1):
                # derive outputs from integrated state, add to solution tensor
                states[:, i:i + 1] = state

            if compute_output:
                output = self.C @ states + self.B @ self.u(time)
                return output

            return states

    def optimize(self, device, target_loader, epochs=1, lr=0.001):
        """
        Optimizes state space matrices via back propagation.

        Args:
            target_data: Dataset containing
            epochs: The number of times the entire dataset is iterated over during training.
            lr: learning rate
            kwargs: Additional keyword arguments to pass to the data_loader that is cre
        """

        loss = []
        opt = self.opt(self.parameters(), lr=lr)
        for i in range(epochs):
            epoch_loss = self.train_one_epoch(device, target_loader, opt)
            loss.append(epoch_loss)
            print(f"Epoch {i} loss: {epoch_loss}")

        return loss


    def train_one_epoch(self, device, data_loader, opt):

        running_loss = 0.0
        for i, (t_i, target_x, target_y, target_dx) in enumerate(data_loader):

            t_i = t_i.to(device)
            target_x = target_x.to(device)
            target_y = target_y.to(device)
            target_dx = target_dx.to(device)

            opt.zero_grad()

            # Run the true states through this model's state space model
            est_dx, est_y = self.forward(t_i, target_x)

            # compute the loss on dx
            dx_loss = nn.functional.mse_loss(est_dx, target_dx)
            dx_loss.backward()

            # compute the loss on output
            out_loss = nn.functional.mse_loss(est_y, target_y)
            out_loss.backward()
            opt.step()

            avg_batch_loss = (out_loss.item() + dx_loss.item()) / 2
            running_loss += avg_batch_loss

        return running_loss / len(data_loader)


if __name__ == '__main__':

    from utils.datagen import generate_spring_data
    from utils.datasets import StateSpaceData
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

    # Generate real-world "sensed" data
    spring_data, A, B, C, D = generate_spring_data(device_type, t0, tn, delta_t, y0, forcing_func, save=True)

    # generate errored state matrices
    error = partial(torch.normal, 0.0, 0.33)
    A = A + error(A.shape)
    B = B + error(B.shape)
    C = C + error(C.shape)
    D = D + error(D.shape)

    # run unoptimized model to get estimated solution.
    est_model = StateSpaceModel(A, B, C, D, forcing_func).to(device_type)
    est_states = est_model.run(device_type, t0, tn, delta_t, y0).detach().to('cpu')

    # Create dataloader and run the optimizer
    spring_loader = DataLoader(spring_data, batch_size=10, shuffle=True)
    est_model.optimize(device_type, spring_loader, epochs=100)

    # re-run the optimized model
    new_est_states = est_model.run(device_type, t0, tn, delta_t, y0, compute_output=False).detach().to('cpu')

    # plot the true and estimated solutions side by side.
    fig, axes = plt.subplots(nrows=4)

    # plot pre-optimization results next to truth data
    t = torch.arange(t0, tn + delta_t, delta_t)

    c = ["cornflowerblue", "wheat"]
    o = [est_states, new_est_states]

    for i, ax in enumerate(axes):

        ix = i % 2

        if ix == 0:
            # plot position information before and after optimization
            ax.plot(t, spring_data["x"][f"x_{0}"], label="sensed position", c="blue")
            if i == 0:
                ax.plot(t, o[0][ix], "--", label="estimated position", c=c[ix])
            else:
                ax.plot(t, o[1][ix], "--", label=" updated position estimate", c=c[ix])
        else:
            # plot velocity information, before and after optimization
            ax.plot(t, spring_data["x"][f"x_{ix}"], label="sensed velocity", c="orange")
            if i == 1:
                ax.plot(t, o[0][ix], "--", label="estimated velocity", c=c[ix])
            else:
                ax.plot(t, o[1][ix], "--", label="updated velocity estimate", c=c[ix])

        ax.legend(loc="lower right")

    plt.tight_layout()
    plt.show()
    #
    # true_params = (A, B, C, D)
    # for p1, p2 in zip(est_model.named_parameters(), true_params):
    #     print(f" Comparison of Matrix {p1[0]}:")
    #     print(p1[1])
    #     print(p2)
    #     print("\n")

"""
Workflow: Simulation 
"""