# Overview

- Pytorch implementation of
  [Self-Attention ConvLSTM for Spatiotemporal Prediction](https://ojs.aaai.org/index.php/AAAI/article/view/6819/6673)
- test on MovingMNIST

## NOTE

- In this repo, I denote `SAM-ConvLSTM` as a model uses
  `self attention memory module` and `SA-ConvLSTM` is a model that uses
  `self attention module`. In the paper, the model uses
  `self attention memory module` is called as `SA-ConvLSTM`.

## Directories

- `models/`
  - `convlstm_cell/`
    - `convlstm_cell.py`
      - ConvLSTM cell implementation based on
        [Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting](https://paperswithcode.com/paper/convolutional-lstm-network-a-machine-learning).
    - `convlstm.py`
      - ConvLSTM implementation.
    - `seq2seq.py`
      - Sequence to sequence model based on ConvLSTM.
  - `self_attention_convlstm/`
    - `self_attention.py`
      - Self Attention module implementation.
    - `sa_convlstm_cell.py`
      - Self Attention ConvLSTM cell implementation based on
        [Self-Attention ConvLSTM for Spatiotemporal Prediction](https://ojs.aaai.org/index.php/AAAI/article/view/6819/6673)
    - `sa_convlstm.py`
      - Self-Attention ConvLSTM implementation.
    - `sa_seq2seq.py`
      - Sequence to sequence model based on Self-Attention ConvLSTM.
  - `self_attention_memory_convlstm/`
    - `self_attention_memory_module.py`
      - Self-Attention memory module implementation based on
        [Self-Attention ConvLSTM for Spatiotemporal Prediction](https://ojs.aaai.org/index.php/AAAI/article/view/6819/6673)
    - `sam_convlstm_cell.py`
      - Self-Attention memory ConvLSTM cell implementation.
    - `sam_convlstm.py`
      - Self-Attention memory ConvLSTM implementation.
    - `sam_seq2seq.py`
      - Sequence to sequence model based on Self-Attention memory ConvLSTM.

## Vizualized Attention Maps

![sa-convlstm](fig/sa-convlstm.png)

The above figure is SAM-ConvLSTM formulation process. alpha_{h} in the figure is
used for visualizing attention maps in evaluation (`evaluate/`). Also see the
following files for all calculation process.

- `models/self_attention_memory_convlstm/sam_convlstm_cell.py`
- `models/self_attention_memory_convlstm/sam_convlstm.py`
- `models/self_attention_memory_convlstm/self_attention_memory_module.py`
