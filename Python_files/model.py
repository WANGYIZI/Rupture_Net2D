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
        # input convolution
        self.conv_input = nn.Conv1d(5, config['conv1_dim'], kernel_size=1, stride=1, padding=0)

        # first convolution block
        self.conv1 = nn.ModuleList()
        cur_dim = config['conv1_dim']
        for i in range(config['conv1_layers']):
            # square dimension in each step for concatination
            self.conv1.append(nn.Conv1d(cur_dim, cur_dim, kernel_size=1, stride=1, padding=0))
            cur_dim = 2 * cur_dim

        # Transformer blocks
        self.attention_layers = nn.ModuleList()
        for i in range(config['num_transformer_blocks']):
            self.attention_layers.append(
                TransformerEncoderBlock(
                    embed_dim=cur_dim,
                    num_heads=config['num_transformer_heads'],
                    dropout_pro=config['dropout_pro'],
                    hidden_dim=config['transformer_hidden_dim']
                )
            )

        # second convolution layers
        self.conv2 = nn.ModuleList()
        self.conv2.append(nn.Conv1d(cur_dim, config['conv2_dim'], kernel_size=1, stride=1, padding=0))

        self.conv_output = nn.Conv1d(config['conv2_dim'], 2, kernel_size=1, stride=1, padding=0)

        self.relu = nn.ReLU()

        # for scheduler
        self.embed_dim = cur_dim

    def forward(self, x):
        # CNN Encoder
        # (N, 100, 5) -> (N, 5, 100)
        x = torch.transpose(x, 1, 2)

        # (N, 5, 100) -> (N, conv1_dim, 100)
        x = self.relu(self.conv_input(x))

        for layer in self.conv1:
            new = self.relu(layer(x))
            x = torch.cat([x, new], dim=1)

        # (N, self.embed_dim, 100) -> (N, 100, self.embed_dim)
        x = torch.transpose(x, 1, 2)

        # Transformer Processor
        for layer in self.attention_layers:
            # (N, 100, self.embed_dim) -> (N, 100, self.embed_dim)
            x = layer(x)

        # CNN decoder
        # (N, 100, self.embed_dim) -> (N, self.embed_dim, 100)
        x = torch.transpose(x, 1, 2)

        # (N, self.embed_dim, 100) -> (N, conv2_dim, 100)
        for layer in self.conv2:
            x = self.relu(layer(x))

        # (N, conv2_dim, 100) -> (N, 2, 100)
        x = self.relu(self.conv_output(x))

        # (N, 2, 100) -> (N, 100, 2)
        x = torch.transpose(x, 1, 2)
        return x
