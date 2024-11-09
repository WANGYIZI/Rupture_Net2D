import torch
import torch.nn as nn


class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout_pro, hidden_dim):
        super(TransformerEncoderBlock, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout_pro = dropout_pro
        self.hidden_dim = hidden_dim
        # Self-Attention
        self.query_projection = nn.Linear(self.embed_dim, self.embed_dim,
                                          bias=False)
        self.key_projection = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.value_projection = nn.Linear(self.embed_dim, self.embed_dim,
                                          bias=False)
        self.attention = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=self.num_heads,
                                               dropout=self.dropout_pro,
                                               batch_first=True)

        # Feed Forward
        self.feed_forward = nn.Sequential(
            nn.Linear(self.embed_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.embed_dim)
        )

        # Layer Normalization
        self.layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(self, x):
        # Self Attention
        residual1 = x
        transpose = x
        query = self.query_projection(transpose)
        key = self.key_projection(transpose)
        value = self.key_projection(transpose)
        attn_output, _ = self.attention(query, key, value)
        sum1 = residual1 + attn_output  # Add & Norm
        layernorm1 = self.layer_norm(sum1)
        # Feed Forward
        residual2 = layernorm1
        feed_forward = self.feed_forward(residual2)
        sum2 = feed_forward + residual2  # Add & Norm
        output = self.layer_norm(sum2)  # Layer Norm
        return output


class RuptureNet2D(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = 8
        self.embed_dim = 1024
        self.dropout_pro = 0.1
        self.hidden_dim = 2048
        self.conv1 = nn.Conv1d(5, 256, kernel_size=1, stride=1, padding=0)  # In conv1d, kernel_size=1, stride=1, padding=0
        self.conv2 = nn.Conv1d(256, 256, kernel_size=1, stride=1, padding=0)

        self.conv4 = nn.Conv1d(512, 512, kernel_size=1, stride=1, padding=0)

        # embeded_dim of Transformer is 1024
        self.SelfAttention1 = TransformerEncoderBlock(embed_dim=1024, num_heads=8, dropout_pro=0.1, hidden_dim=2048)
        self.SelfAttention2 = TransformerEncoderBlock(embed_dim=1024, num_heads=8, dropout_pro=0.1, hidden_dim=2048)
        self.SelfAttention3 = TransformerEncoderBlock(embed_dim=1024, num_heads=8, dropout_pro=0.1, hidden_dim=2048)
        self.SelfAttention4 = TransformerEncoderBlock(embed_dim=1024, num_heads=8, dropout_pro=0.1, hidden_dim=2048)
        self.SelfAttention5 = TransformerEncoderBlock(embed_dim=1024, num_heads=8, dropout_pro=0.1, hidden_dim=2048)
        self.SelfAttention6 = TransformerEncoderBlock(embed_dim=1024, num_heads=8, dropout_pro=0.1, hidden_dim=2048)
        self.relu = nn.ReLU()
        self.conv7 = nn.Conv1d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.conv8 = nn.Conv1d(256, 2, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # CNN encoder
        transpose1 = torch.transpose(x, 1, 2)
        conv1 = self.relu(self.conv1(transpose1))
        conv2 = self.relu(self.conv2(conv1))
        cat1 = torch.cat([conv1, conv2], dim=1)
        conv4 = self.relu(self.conv4(cat1))
        cat2 = torch.cat([cat1, conv4], dim=1)
        transpose2 = torch.transpose(cat2, 1, 2)

        # Transformer Processor
        #  1th encoder_block
        block1 = self.SelfAttention1(transpose2)
        # 2nd encoder_block
        block2 = self.SelfAttention2(block1)
        # 3rd encoder_block
        block3 = self.SelfAttention3(block2)
        # 4th encoder_block
        block4 = self.SelfAttention4(block3)
        # 5th encoder_block
        block5 = self.SelfAttention5(block4)
        # 6th encoder_block
        block6 = self.SelfAttention6(block5)

        # CNN decoder
        x = torch.transpose(block6, 1, 2)
        conv7 = self.relu(self.conv7(x))
        conv8 = self.conv8(conv7)
        Front_Slip = torch.transpose(conv8, 1, 2)
        output = self.relu(Front_Slip)
        return output
