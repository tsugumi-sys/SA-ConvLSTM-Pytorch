from torch import nn
from torch.optim import Adam

from core.constants import WeightsInitializer
from data_loaders.moving_mnist import MovingMNISTDataLoaders
from pipelines.experimenter import Experimenter
from pipelines.trainer import TrainingParams
from pipelines.utils.early_stopping import EarlyStopping
from self_attention_memory_convlstm.seq2seq import SAMSeq2Seq, SAMSeq2SeqParams


def main():
    ###
    # Common Params
    ###
    artifact_dir = "./tmp"
    input_seq_length = 10
    train_batch_size = 32
    validation_bath_size = 16
    ###
    # Setup Pipeline
    ###
    model_params: SAMSeq2SeqParams = {
        "attention_hidden_dims": 2,
        "input_seq_length": input_seq_length,
        "num_layers": 2,
        "num_kernels": 64,
        "return_sequences": False,
        "convlstm_params": {
            "in_channels": 1,
            "out_channels": 1,
            "kernel_size": (3, 3),
            "padding": "same",
            "activation": "relu",
            "frame_size": (64, 64),
            "weights_initializer": WeightsInitializer.He,
        },
    }

    model = SAMSeq2Seq(**model_params)

    training_params: TrainingParams = {
        "epochs": 1,
        "loss_criterion": nn.BCELoss(reduction="sum"),
        "accuracy_criterion": nn.L1Loss(),
        "optimizer": Adam(model.parameters(), lr=1e-4),
        "early_stopping": EarlyStopping(
            patience=30,
            verbose=True,
            delta=0.0001,
        ),
        "metrics_filename": "metrics.csv",
    }

    print("Loading dataset ...")
    data_loaders = MovingMNISTDataLoaders(
        train_batch_size=train_batch_size,
        validation_batch_size=validation_bath_size,
        input_frames=model_params["input_seq_length"],
        label_frames=1,
        split_ratios=[0.7, 0.299, 0.001],
    )

    experimenter = Experimenter(artifact_dir, data_loaders, model, training_params)
    experimenter.run()


if __name__ == "__main__":
    main()
