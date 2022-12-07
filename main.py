import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision.utils import save_image
from torchvision import transforms

from dataset.moving_mnist import MovingMNIST
from models.convlstm.seq2seq import Seq2Seq
from models.self_attention_convlstm.sa_seq2seq import SASeq2Seq
from models.self_attention_memory_convlstm.sam_seq2seq import SAMSeq2Seq


def main():
    train_epochs = 500
    train_batch_size = 8

    attention_hidden_dims = 4
    num_channels = 1
    kernel_size = 3
    num_kernels = 64
    padding = "same"
    activation = "relu"
    frame_size = (64, 64)
    num_layers = 4
    input_seq_length = 10
    weights_initializer = "he"
    return_sequences = True

    loss = nn.MSELoss(reduction="sum")

    moving_mnist = MovingMNIST(root="./data", input_seq_length=10, download=True)
    train_dataset, valid_dataset, test_dataset = random_split(
        moving_mnist, [0.7, 0.299, 0.001], generator=torch.Generator().manual_seed(42)
    )

    train_dataloader = DataLoader(
        train_dataset, batch_size=train_batch_size, shuffle=True
    )
    valid_dataloader = DataLoader(valid_dataset, batch_size=len(valid_dataset))

    sample_dataset, label = test_dataset.__getitem__(0)
    transform = transforms.ToPILImage()
    for seq_idx in range(10):
        seq = sample_dataset[:, seq_idx]
        pil_img = transform(seq)
        pil_img.save(f"img{seq_idx}.png")
        # save_image(pil_img, f"img{seq_idx}.png")


if __name__ == "__main__":
    main()
