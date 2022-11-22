import sys
from typing import Tuple

import torch
from torch import nn

sys.path.append("..")
from train.src.config import DEVICE


class SelfAttentionMemoryWithConv2d(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super(SelfAttentionMemoryWithConv2d, self).__init__()

        # attention for hidden layer
        self.query_h = nn.Conv2d(input_dim, hidden_dim, 1)
        self.key_h = nn.Conv2d(input_dim, hidden_dim, 1)
        self.value_h = nn.Conv2d(input_dim, input_dim, 1)
        self.z_h = nn.Conv2d(input_dim, input_dim, 1)

        # attention for memory layer
        self.key_m = nn.Conv2d(input_dim, hidden_dim, 1)
        self.value_m = nn.Conv2d(input_dim, input_dim, 1)
        self.z_m = nn.Conv2d(input_dim, input_dim, 1)

        # weights of concated channels of h Zh and Zm.
        self.w_z = nn.Conv2d(input_dim * 2, input_dim * 2, 1)

        # weights of conated channels of Z and h.
        self.w = nn.Conv2d(input_dim * 3, input_dim * 3, 1)

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

    def forward(self, h, m) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return:
            Tuple(tirch.Tensor, torch.Tensor): new Hidden layer and new memory module.
        """
        batch_size, _, H, W = h.shape
        # hidden attention
        k_h = self.key_h(h)
        q_h = self.query_h(h)
        v_h = self.value_h(h)

        k_h = k_h.view(batch_size, self.hidden_dim, H * W)
        q_h = q_h.view(batch_size, self.hidden_dim, H * W).transpose(1, 2)
        v_h = v_h.view(batch_size, self.hidden_dim, H * W)

        attention_h = torch.softmax(torch.bmm(q_h, k_h), dim=-1)  # The shape is (batch_size, H*W, H*W)
        z_h = torch.matmul(attention_h, v_h.permute(0, 2, 1))
        z_h = z_h.transpose(1, 2).view(batch_size, self.input_dim, H, W)

        # memotry attention
        k_m = self.key_m(m)
        v_m = self.value_m(m)

        k_m = k_m.view(batch_size, self.hidden_dim, H * W)
        v_m = v_m.view(batch_size, self.hidden_dim, H * W)

        attention_m = torch.softmax(torch.bmm(q_h, k_m), dim=-1)
        z_m = torch.matmul(attention_m, v_m.permute(0, 2, 1))
        z_m = z_m.transpose(1, 2).view(batch_size, self.input_dim, H, W)

        # channel concat of Zh and Zm.
        Z = torch.cat([z_h, z_m], dim=1)
        Z = self.w_z(Z)

        # channel concat of Z and h
        W = torch.cat([Z, h], dim=1)
        W = self.w(W)

        # mi_conv: Wm;zi * Z + Wm;hi * Ht + bm;i
        # mg_conv: Wm;zg * Z + Wm;hg * Ht + bm;g
        # mo_conv: Wm;zo * Z + Wm;ho * Ht + bm;o
        mi_conv, mg_conv, mo_conv = torch.chunk(W, chunks=self.input_dim, dim=1)
        input_gate = torch.sigmoid(mi_conv)
        g = torch.tanh(mg_conv)
        new_M = (1 - input_gate) * m + input_gate * g
        output_gate = torch.sigmoid(mo_conv)
        new_H = output_gate * new_M

        return new_H, new_M
