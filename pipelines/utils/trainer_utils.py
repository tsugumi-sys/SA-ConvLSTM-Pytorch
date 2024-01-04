import torch
from torch import nn

# TODO: consider place saving function into each models directory.
from convlstm.seq2seq import Seq2Seq
from self_attention_convlstm.seq2seq import SASeq2Seq
from self_attention_memory_convlstm.seq2seq import SAMSeq2Seq


def save_seq2seq_model(model: nn.Module, save_path: str) -> None:
    if isinstance(model, Seq2Seq):
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "num_channels": model.num_channels,
                "kernel_size": model.kernel_size,
                "num_kernels": model.num_kernels,
                "padding": model.padding,
                "activation": model.activation,
                "frame_size": model.frame_size,
                "num_layers": model.num_layers,
                "weights_initializer": model.weights_initializer,
            },
            save_path,
        )
    elif isinstance(model, (SASeq2Seq, SAMSeq2Seq)):
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "attention_hidden_dims": model.attention_hidden_dims,
                "num_channels": model.num_channels,
                "kernel_size": model.kernel_size,
                "num_kernels": model.num_kernels,
                "padding": model.padding,
                "activation": model.activation,
                "frame_size": model.frame_size,
                "num_layers": model.num_layers,
                "weights_initializer": model.weights_initializer,
            },
            save_path,
        )
    else:
        raise ValueError(f"Unknown model {model}")
    return
