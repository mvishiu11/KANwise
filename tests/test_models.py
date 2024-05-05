import pandas as pd
import pytest
import torch

from kanwise.utils import CustomDataset
from kanwise.utils import get_dataloader
from kanwise.utils import load_data


def test_load_data(tmp_path):
    """
    Test the load_data function.
    """
    # Create a dummy CSV file
    d = tmp_path / "sub"
    d.mkdir()
    p = d / "data.csv"
    data = "col1,col2,target\n1,2,0\n3,4,1"
    p.write_text(data)

    # Test load_data
    df = load_data(p)
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (2, 3)


def test_custom_dataset():
    """
    Test the CustomDataset class.
    """
    data = pd.DataFrame({"col1": [1, 3], "col2": [2, 4], "target": [0, 1]})
    dataset = CustomDataset(data)
    x, y = dataset[0]
    assert torch.equal(x, torch.tensor([1.0, 2.0]))
    assert y == 0


def test_get_dataloader(tmp_path):
    """
    Test the get_dataloader function.
    """
    # Use the same setup as test_load_data
    d = tmp_path / "sub"
    d.mkdir()
    p = d / "data.csv"
    data = "col1,col2,target\n1,2,0\n3,4,1"
    p.write_text(data)

    loader = get_dataloader(p, batch_size=1)
    for batch in loader:
        x, y = batch
        assert x.shape == (1, 2)  # batch size, number of features
        assert y.shape == (1,)  # batch size


@pytest.mark.parametrize(
    "path,expected",
    [
        ("path/to/nonexistent/file.csv", FileNotFoundError),
        (123, ValueError),  # Updated to expect ValueError which is thrown by pandas
    ],
)
def test_load_data_exceptions(path, expected):
    with pytest.raises(expected):
        load_data(path)
