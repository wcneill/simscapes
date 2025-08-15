import pandas as pd
from torch.utils.data import Dataset
import torch


class StateSpaceData(Dataset):

    """
    A dataset designed to handle wide format tensor data.

    This WideTensorData accepts wide-format data stored a tensor. When indexing into the
    dataset, a tuple is returned. For example, if a single divide is passed, a two-tuple is returned,
    where the first item in the tuple is all columns for the given index up
    to dividing_index. The second item contains data in all columns from dividing_index to the end.

    Any number of divides may be passed up to n - 1 indices, where n is the number of columns or rows in the
    wide or long data respectively.

    Args:
        tensor: Wide format data stored in a torch tensor
        dividing_index: An integer or list of integers - The index(s) along which to divide the data.
            in "wide" arrangement, this splits the data by colum. in "long" the data is split by row.
    """
    def __init__(self, df=None, file_loc=None, transform=None):
        super().__init__()
        if file_loc is not None:
            self.data = pd.read_csv(file_loc, header=[0, 1], index_col=0)
        elif df is not None:
            self.data=df
        else:
            raise RuntimeError("Must either provide an in memory DataFrame, or read one from CSV. ")

        self.tensor_names = []
        for name in self.data.columns.get_level_values(0):
            if name not in self.tensor_names:
                self.tensor_names.append(name)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if isinstance(idx, str):
            return self.data[idx]

        returns = []
        for name in self.tensor_names:
            tensor_data = torch.tensor(self.data.iloc[idx][name].to_numpy())
            tensor_data = torch.atleast_2d(tensor_data)
            returns.append(tensor_data.T)

        return tuple(returns)
