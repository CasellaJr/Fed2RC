import torch
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, StandardScaler
from fluke.data import DataContainer


def UCRArchive(path: str = "../data",
               ds_name: str = "Adiac") -> DataContainer:
    """Load the UCRArchive dataset.
    The dataset is split into training and testing sets according to the default split of the
    UCR Archive.
    The archive contains 128 datasets.
    Reference: https://www.cs.ucr.edu/%7Eeamonn/time_series_data_2018/

    Args:
        path (str, optional): The path where the dataset is stored. Defaults to ``"../data"``.
        transforms (callable, optional): The transformations to apply to the data. Defaults to
            ``None``.

    Returns:
        DataContainer: The UCRArchive datasets.
    """
    LE = LabelEncoder()
    train_data = pd.read_table(
        f'{path}/UCRArchive_2018/{ds_name}/{ds_name}_TRAIN.tsv', header=None).to_numpy()
    X_train = train_data[:, 1:]
    Y_train = train_data[:, 0]
    Y_train = LE.fit_transform(Y_train)
    test_data = pd.read_table(
        f'{path}/UCRArchive_2018/{ds_name}/{ds_name}_TEST.tsv', header=None).to_numpy()
    X_test = test_data[:, 1:]
    Y_test = test_data[:, 0]
    Y_test = LE.transform(Y_test)

    # Standardize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train = torch.tensor(X_train.astype(np.float32))
    Y_train = torch.LongTensor(Y_train.astype(np.int32))
    X_test = torch.tensor(X_test.astype(np.float32))
    Y_test = torch.LongTensor(Y_test.astype(np.int32))

    return DataContainer(X_train,
                         Y_train,
                         X_test,
                         Y_test,
                         len(np.unique(Y_train)),
                         None)
