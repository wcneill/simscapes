import torch
from torch import optim
from spring import StateSpaceModel
from utils.datasets import StateSpaceData

import pandas as pd

def generate_spring_data(device, t0, tn, dt, y0, forcing_function, save=True, file_name="spring_data.csv"):
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

    with torch.no_grad():
        # run the underlying integrator to generate solution data to use in training.
        states = true_model.run(device, t0, tn, dt, y0)

        # Plug solution back in to ODE
        time = torch.arange(t0, tn + dt, dt).to(device)
        dx, outputs = true_model.forward(time, states)

    # Generate dataframe to contain data
    tensor_map = {
        "t": time,
        "x": states,
        "y": outputs,
        "dx": dx
    }

    # Define levels of column index
    top_level = []
    sub_level = []

    for name, tensor in tensor_map.items():
        num_subcols = tensor.shape[0] if tensor.ndim > 1 else 1
        top_level.extend([name] * num_subcols)
        sub_level.extend([f"{name}_{i}" for i in range(num_subcols)])  # subcolumn labels as strings

    # Create multi-index object from definition of columns
    # Step 2: Create MultiIndex
    columns = pd.MultiIndex.from_arrays([top_level, sub_level])

    data = torch.vstack([t if t.ndim > 1 else t.unsqueeze(0) for t in tensor_map.values()])
    df = pd.DataFrame(data.T.cpu(), columns=columns)

    if save:
        df.to_csv(file_name)

    return StateSpaceData(df=df), A, B, C, D

