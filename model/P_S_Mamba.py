import torch
import torch.nn as nn
from layers.Mamba_EncDec import Encoder, EncoderLayer
from layers.Embed import PatchEmbedding
from mamba_ssm import Mamba

class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [batch_size, nvars, d_model, patch_num]
        x = self.flatten(x)  # Flatten to [batch_size, nvars, d_model * patch_num]
        x = self.linear(x)    # Linear layer to map to target_window
        x = self.dropout(x)   # Dropout for regularization
        return x

class Model(nn.Module):
    """
    S-Mamba model with overlapping patch input
    """

    def __init__(self, configs, stride=8):
        super(Model, self).__init__()
        # Store configurations for sequence and prediction lengths, and model behavior
        self.seq_len = configs.seq_len           # Input sequence length
        self.pred_len = configs.pred_len          # Output prediction length
        self.output_attention = configs.output_attention  # Whether to output attention weights
        self.use_norm = configs.use_norm          # Whether to normalize the inputs

        # Calculate patch number dynamically as in C-Mamba
        self.patch_len = getattr(configs, "patch_len", 16)  # Set a default patch length if not provided
        self.stride = getattr(configs, "stride", stride)
        self.patch_num = int((self.seq_len - self.patch_len) / self.stride + 2)

        # Initialize PatchEmbedding using C-Mamba’s implementation
        self.enc_embedding = PatchEmbedding(
            configs.d_model, self.patch_len, self.stride, self.stride, configs.dropout
        )

        # Encoder-only architecture
        self.encoder = Encoder(
            [
                # Create encoder layers, each with two Mamba blocks and feedforward network
                EncoderLayer(
                    Mamba(
                        d_model=configs.d_model,    # Model dimension
                        d_state=configs.d_state,    # State dimension in Mamba block
                        d_conv=2,                   # Convolution layer dimension
                        expand=1                    # Expansion factor
                    ),
                    Mamba(
                        d_model=configs.d_model,
                        d_state=configs.d_state,
                        d_conv=2,
                        expand=1
                    ),
                    d_model=configs.d_model,        # Model dimension
                    d_ff=configs.d_model * 4,       # Feedforward network dimension
                    dropout=configs.dropout,        # Dropout rate
                    activation=configs.activation   # Activation function
                ) for _ in range(configs.e_layers)  # Repeat for specified number of encoder layers
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)  # Layer normalization for stable training
        )

        # Prediction Head using a Flattening approach like in C-Mamba
        self.head_nf = configs.d_model * self.patch_num
        self.head = FlattenHead(configs.enc_in, self.head_nf, self.pred_len, head_dropout=configs.dropout)

    # Forecasting function to generate predictions
    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Optional instance normalization on the input sequence
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        # Transpose to [batch_size, features, sequence_length] to match PatchEmbedding input requirements
        x_enc = x_enc.permute(0, 2, 1)

        # Apply patch embedding and retrieve enc_out from the first element of the tuple
        # enc_out: [batch_size * num_vars, patch_num, d_model]
        enc_out, n_vars = self.enc_embedding(x_enc)

        # Pass through encoder and extract the primary output if it's returning a tuple
        enc_out = self.encoder(enc_out)
        if isinstance(enc_out, tuple):
            enc_out = enc_out[0]  # Extract only the encoded output tensor if it's a tuple

        # Reshape back to [batch_size, num_vars, d_model, patch_num]
        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))

        # Transpose for the final head processing
        enc_out = enc_out.permute(0, 1, 3, 2)  # Shape: [batch_size, num_vars, patch_num, d_model]

        # Prediction head to generate the final output
        dec_out = self.head(enc_out)  # Shape: [batch_size, num_vars, target_window]
        dec_out = dec_out.permute(0, 2, 1)  # Transpose to [batch_size, target_window, num_vars]

        # De-normalize if instance normalization was applied
        if self.use_norm:
            # Expand or repeat stdev to match dec_out's shape
            stdev = stdev.expand(-1, self.pred_len, -1)  # Repeat along the sequence length dimension
            means = means.expand(-1, self.pred_len, -1)  # Repeat along the sequence length dimension

            # Apply inverse normalization
            dec_out = dec_out * stdev
            dec_out = dec_out + means

        return dec_out

    # Main forward function for training and inference
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # Generate forecasted predictions
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        # Return last `pred_len`
        return dec_out[:, -self.pred_len:, :]  # Shape: [Batch, pred_len, Features]
