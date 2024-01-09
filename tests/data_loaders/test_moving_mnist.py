from unittest.mock import patch

from data_loaders.moving_mnist import MovingMNISTDataLoaders
from tests.utils import MockMovingMNIST


@patch("data_loaders.moving_mnist.MovingMNIST")
def test_MovingMNISTDataLoaders(mocked_MovingMNIST):
    input_frames = 10
    dataset_length = 10
    mocked_MovingMNIST.return_value = MockMovingMNIST(
        dataset_length=dataset_length, split_ratio=input_frames
    )
    dataloaders = MovingMNISTDataLoaders(train_batch_size=1, input_frames=input_frames)
    assert len(dataloaders.train_dataloader) == 7
    assert len(dataloaders.validation_dataloader) == 2
    assert len(dataloaders.test_dataloader) == 1
    # print(next(iter(dataloaders.train_dataloader)))
    for _, (input, target) in enumerate(dataloaders.train_dataloader, start=1):
        print(input, target)
