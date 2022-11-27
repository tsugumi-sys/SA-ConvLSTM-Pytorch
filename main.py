from models.convlstm.seq2seq import Seq2Seq
from models.self_attention_convlstm.sa_seq2seq import SASeq2Seq
from models.self_attention_memory_convlstm.sam_seq2seq import SAMSeq2Seq


def main():
    train_epochs = 500
    mnist_dataset = 