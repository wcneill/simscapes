from torch.utils.data import Dataset
import torch


class TensorData(Dataset):

    """
    A dataset designed to handle wide format tensor data.

    This WideTensorData accepts wide-format data stored a tensor. When indexing into the
    dataset, a two-tuple is returned. The first item in the tuple is all columns for the given index up
    to dividing_index. The second item contains data in all columns from dividing_index to the end.

    Args:
        tensor: Wide format data stored in a torch tensor
        dividing_index: The column index along to divide the data to be returned.
    """
    def __init__(self, tensor, dividing_index, arrangement="wide", transform=None):
        super().__init__()
        self.data = tensor
        self.divide = dividing_index
        self.arrangement = arrangement

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.arrangement == "wide":
            first = self.data[idx, :self.divide]
            second = self.data[idx, self.divide:]
        elif self.arrangement == "long":
            first = self.data[:self.divide, idx:idx + 1]
            second = self.data[self.divide:, idx:idx + 1]
        else:
            RuntimeError("Unrecognized format")

        return first, second
