from torch import nn
from torch.optim import Adam

from pipeline.data_loader.moving_mnist import MovingMNISTDataLoaders
from pipeline.evaluator import Evaluator
from pipeline.trainer import Trainer
from pipeline.utils.early_stopping import EarlyStopping

# from models.convlstm.seq2seq import Seq2Seq
# from models.self_attention_convlstm.sa_seq2seq import SASeq2Seq
from self_attention_memory_convlstm.seq2seq import SAMSeq2Seq


def main():
    ###
    # Parameters
    ###
    train_epochs = 1
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
    out_channels = 1

    ###
    # Dataset and DataLoader
    ###
    data_loaders = MovingMNISTDataLoaders(
        train_batch_size, input_frames=input_seq_length
    )

    ###
    # Model, Loss function and Optimizer, e.t.c
    ###
    loss_criterion = nn.MSELoss()
    acc_criterion = nn.L1Loss()
    model = SAMSeq2Seq(
        attention_hidden_dims=attention_hidden_dims,
        num_channels=num_channels,
        kernel_size=kernel_size,
        num_kernels=num_kernels,
        padding=padding,
        activation=activation,
        frame_size=frame_size,
        num_layers=num_layers,
        input_seq_length=input_seq_length,
        out_channels=out_channels,
        weights_initializer=weights_initializer,
        return_sequences=return_sequences,
    )
    optimizer = Adam(model.parameters(), lr=0.0005)
    early_stopping = EarlyStopping(
        patience=30,
        verbose=True,
        delta=0.0001,
    )

    ###
    # Train
    ###
    trainer = Trainer(
        save_model_path="./tmp",
        model=model,
        train_epochs=train_epochs,
        train_dataloader=data_loaders.train_dataloader,
        valid_dataloader=data_loaders.valid_dataloader,
        loss_criterion=loss_criterion,
        accuracy_criterion=acc_criterion,
        optimizer=optimizer,
        early_stopping=early_stopping,
        save_metrics_path="./tmp",
    )
    trainer.run()

    ###
    # Evaluation
    ###
    evaluator = Evaluator(
        model=None,
        test_dataloader=data_loaders.test_dataloader,
        save_dir_path="./tmp/evaluate",
    )
    evaluator.run()


if __name__ == "__main__":
    main()
