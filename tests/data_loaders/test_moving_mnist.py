from unittest.mock import patch

import pytest

from data_loaders.moving_mnist import MovingMNISTDataLoaders
from tests.utils import MockMovingMNIST


@patch("data_loaders.moving_mnist.MovingMNIST")
def test_MovingMNISTDataLoaders(mocked_MovingMNIST):
    dataset_length = 10
    train_batch_size = 2
    input_frames = 10
    mocked_MovingMNIST.return_value = MockMovingMNIST(dataset_length=dataset_length)
    dataloaders = MovingMNISTDataLoaders(
        train_batch_size=train_batch_size, input_frames=input_frames
    )
    assert len(dataloaders.train_dataloader) == 4
    assert len(dataloaders.validation_dataloader) == 2
    assert len(dataloaders.test_dataloader) == 1
    input, target = next(iter(dataloaders.train_dataloader))
    assert input.size(0) == train_batch_size
    assert input.size(2) == input_frames
    assert target.size(0) == train_batch_size
    assert target.size(2) == input_frames


@patch("data_loaders.moving_mnist.MovingMNIST")
def test_MovingMNISTDataLoaders_label_frames_set(mocked_MovingMNIST):
    dataset_length = 10
    train_batch_size = 2
    input_frames = 10
    label_frames = 1
    mocked_MovingMNIST.return_value = MockMovingMNIST(dataset_length=dataset_length)
    dataloaders = MovingMNISTDataLoaders(
        train_batch_size=train_batch_size,
        input_frames=input_frames,
        label_frames=label_frames,
    )
    assert len(dataloaders.train_dataloader) == 4
    assert len(dataloaders.validation_dataloader) == 2
    assert len(dataloaders.test_dataloader) == 1
    input, target = next(iter(dataloaders.train_dataloader))
    assert input.size(0) == train_batch_size
    assert input.size(2) == input_frames
    assert target.size(0) == train_batch_size
    assert target.size(2) == label_frames


@patch("data_loaders.moving_mnist.MovingMNIST")
def test_MovingMNISTDataLoaders_split_ratio_set(mocked_MovingMNIST):
    dataset_length = 10
    train_batch_size = 1
    input_frames = 10

    with pytest.raises(ValueError):
        dataloaders = MovingMNISTDataLoaders(
            train_batch_size=train_batch_size,
            input_frames=input_frames,
            split_ratios=(0.6, 0.1, 0.1),
        )

    with pytest.raises(ValueError):
        dataloaders = MovingMNISTDataLoaders(
            train_batch_size=train_batch_size,
            input_frames=input_frames,
            split_ratios=(0.6, 0.3, 0.3),
        )

    mocked_MovingMNIST.return_value = MockMovingMNIST(dataset_length=dataset_length)
    dataloaders = MovingMNISTDataLoaders(
        train_batch_size=train_batch_size,
        input_frames=input_frames,
        split_ratios=(0.6, 0.2, 0.2),
    )
    assert len(dataloaders.train_dataloader) == 6
    assert len(dataloaders.validation_dataloader) == 2
    assert len(dataloaders.test_dataloader) == 2
