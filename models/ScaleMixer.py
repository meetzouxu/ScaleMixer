import torch
import torch.nn as nn
import torch.fft


from layers.Embed import PatchEmbedding
from layers.RevIN import RevIN


def FFT_for_Period(x, k=2):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


def select_patch_len(period_list):
    possible_patch_lens = [2, 4, 6, 8, 12, 16, 20, 24]
    selected_patch_lens = [min(possible_patch_lens, key=lambda x: abs(x - period)) for period in period_list]
    return selected_patch_lens


class Flatten_Head(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class MLPBlock(nn.Module):
    def __init__(self, input_dim, mlp_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, mlp_dim)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(mlp_dim, input_dim)

    def forward(self, x):
        # [B, L, D] or [B, D, L]
        return self.fc2(self.gelu(self.fc1(x)))


class TemporalMixing(nn.Module):
    def __init__(self, input_dim, mlp_dim, sampling):
        super().__init__()

        assert sampling in [1, 2, 3, 4, 6, 8, 12]
        self.sampling = sampling
        self.temporal_fac = nn.ModuleList([
            MLPBlock(input_dim // sampling, mlp_dim) for _ in range(sampling)
        ])

    def merge(self, shape, x_list):
        y = torch.zeros(shape, device=x_list[0].device)
        for idx, x_pad in enumerate(x_list):
            y[:, :, idx::self.sampling] = x_pad

        return y

    def forward(self, x):
        x_samp = []
        for idx, samp in enumerate(self.temporal_fac):
            x_samp.append(samp(x[:, :, idx::self.sampling]))

        x = self.merge(x.shape, x_samp)

        return x


class ChannelMixing(nn.Module):
    def __init__(self, input_dim, factorized_dim):
        super().__init__()

        # assert input_dim > factorized_dim
        self.channel_mixing = MLPBlock(input_dim, factorized_dim)

    def forward(self, x):
        return self.channel_mixing(x)


class MLPMixerBlock(nn.Module):
    """
    tokens_dim: d_model
    channels_dim: patch_num
    tokens_hidden_dim: d_model
    channels_hidden_dim: patch_num
    """
    def __init__(self, tokens_dim, channels_dim, tokens_hidden_dim, channels_hidden_dim, sampling):
        super().__init__()
        self.tokens_mixing = TemporalMixing(tokens_dim, tokens_hidden_dim, sampling)
        # self.tokens_mixing = MLPBlock(tokens_dim, tokens_hidden_dim)
        self.channels_mixing = ChannelMixing(channels_dim, channels_hidden_dim)
        # self.channels_mixing = MLPBlock(channels_dim, channels_hidden_dim)
        self.norm = nn.LayerNorm(tokens_hidden_dim)

    def forward(self, x):  # x: [bs * nvars, patch_num, d_model]
        # token-mixing [B, D, #tokens]
        y = self.norm(x) if self.norm else x
        y = self.tokens_mixing(y)

        # channel-mixing [B, #tokens, D]
        # y += x
        # res = y
        # y = self.norm(y) if self.norm else y
        # y = res + self.channels_mixing(y.transpose(1, 2)).transpose(1, 2)

        return y


class ScaleMixerBlock(nn.Module):
    def __init__(self, configs, period_lens, period_weight):
        super().__init__()
        self.configs = configs
        self.patch_lens = period_lens
        self.strides = [patch_len // 2 for patch_len in self.patch_lens]
        self.period_weight = period_weight
        self.layer = configs.e_layers
        self.patch_nums = [int((configs.seq_len - patch_len) / stride + 2) for patch_len, stride in
                      zip(self.patch_lens, self.strides)]
        
        self.mixer_blocks = nn.ModuleList([
            MLPMixerBlock(configs.d_model, patch_num, configs.d_model, patch_num, configs.sampling)
            for patch_num in self.patch_nums
        ])
        
        self.patch_embeddings = nn.ModuleList([
            PatchEmbedding(configs.d_model, patch_len, stride, configs.dropout) for patch_len, stride in
            zip(self.patch_lens, self.strides)
        ])
    
    def forward(self, x):
        outputs = [] 
        for patch_embedding, mlp_mixer in zip(self.patch_embeddings, self.mixer_blocks):
            x_patch_emb, _ = patch_embedding(x)# [bs*nvars, patch_num, d_model]
            x_out = mlp_mixer(x_patch_emb)
            outputs.append(x_out)
        stacked_outputs = torch.stack(outputs, dim=0)  # [len(output), bs*nvars, patch_num, d_model]

        adjusted_weights = self.period_weight.view(-1, 1, 1, 1).to(stacked_outputs.device) # [len(output), 1, 1, 1]
        combined = (stacked_outputs * adjusted_weights).sum(dim=0)

        return combined


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.k = configs.fft_k
        self.layer = configs.e_layers

        self.layer_norm = nn.LayerNorm(configs.d_model)
        self.revin = RevIN(configs.enc_in)

        self.period_list, self.period_weight = FFT_for_Period(torch.randn(1, configs.seq_len, configs.enc_in), configs.fft_k)
        self.patch_lens = select_patch_len(self.period_list)
        self.strides = [patch_len // 2 for patch_len in self.patch_lens]

        self.patch_nums = [int((configs.seq_len - patch_len) / stride + 2) for patch_len, stride in
                      zip(self.patch_lens, self.strides)]

        self.scale_mixer_blocks = nn.ModuleList([
            ScaleMixerBlock(configs,  self.patch_lens, self.period_weight)
            for _ in range(1)
        ])

        # self.patch_embeddings = nn.ModuleList([
        #     PatchEmbedding(configs.d_model, patch_len, stride, configs.dropout) for patch_len, stride in
        #     zip(self.patch_lens, self.strides)
        # ])

        self.head = Flatten_Head(configs.enc_in, configs.d_model * self.patch_nums[-1], configs.pred_len,
                                 head_dropout=configs.dropout)
        self.comb = nn.Linear(configs.e_layers, 1)

    def forecast(self, x):  # x: [bs, seq_len, nvars]
        x = self.revin(x, 'norm')
        x = x.permute(0, 2, 1)  # [bs, nvars, seq_len]
        
        for scale_mixer_block in self.scale_mixer_blocks: # [bs*nvars, patch_num, d_model]
            x = scale_mixer_block(x)

        x = torch.reshape(x, (-1, self.configs.enc_in, x.shape[-2], x.shape[-1]))  # [bs, nvars, patch_num, d_model]
        x = x.permute(0, 1, 3, 2)  # [bs, nvars, d_model, patch_num]

        x = self.head(x)  # [bs, nvars, target_window]
        x = x.permute(0, 2, 1)  # [bs, target_window, nvars]
        x = self.revin(x, 'denorm')

        return x

    def forward(self, x_input, x_mark_input, dec_inp, batch_y_mark, mask=None):
        out = self.forecast(x_input)
        return out  # [B, L, C]
