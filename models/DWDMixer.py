import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Autoformer_EncDec import series_decomp
from layers.Embed import DataEmbedding_wo_pos
from layers.StandardNorm import Normalize
from pytorch_wavelets import DWT1DForward, DWT1DInverse
from torch.nn import TransformerEncoder, TransformerEncoderLayer

import pywt

class DFT_series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, top_k=5):
        super(DFT_series_decomp, self).__init__()
        self.top_k = top_k

    def forward(self, x):
        xf = torch.fft.rfft(x)
        freq = abs(xf)
        freq[0] = 0
        top_k_freq, top_list = torch.topk(freq, self.top_k)
        xf[freq <= top_k_freq.min()] = 0
        x_season = torch.fft.irfft(xf)
        x_trend = x - x_season
        return x_season, x_trend
# for other wavelet bases    
# def create_wavelet_filter(wave: str, in_channels: int):
#     """
#     Create wavelet decomposition and reconstruction filters for 1D conv.
#     Returns:
#         dec_filters: [2*in_channels, 1, K]
#         rec_filters: [2*in_channels, 1, K]
#     """
#     w = pywt.Wavelet(wave)
#     # Reverse for convolution
#     dec_lo = torch.tensor(w.dec_lo[::-1], dtype=torch.float32)  # low-pass
#     dec_hi = torch.tensor(w.dec_hi[::-1], dtype=torch.float32)  # high-pass
#     rec_lo = torch.tensor(w.rec_lo, dtype=torch.float32)        # already in correct order
#     rec_hi = torch.tensor(w.rec_hi, dtype=torch.float32)
#     base_dec = torch.stack([dec_lo, dec_hi], dim=0).unsqueeze(1)   # [2, 1, K]
#     base_rec = torch.stack([rec_lo, rec_hi], dim=0).unsqueeze(1)
#     dec_filters = base_dec.repeat(in_channels, 1, 1)  # [2*C, 1, K]
#     rec_filters = base_rec.repeat(in_channels, 1, 1)  # [2*C, 1, K]
#     return dec_filters, rec_filters

# def wavelet_transform(x, dec_filters):
#     """
#     x: [B, C, L]
#     dec_filters: [2*C, 1, K]
#     Returns:
#         approx: [B, C, L//2]
#         detail: [B, C, L//2]
#     """
#     B, C, L = x.shape
#     K = dec_filters.shape[-1]
#     pad_len = K - 1
#     # Symmetric padding: [pad_left, pad_right]
#     x_padded = F.pad(x, (pad_len // 2, pad_len - pad_len // 2), mode='reflect')
#     out = F.conv1d(x_padded, dec_filters.to(x.device), stride=2, groups=C)  # [B, 2C, L//2]
#     approx, detail = out.chunk(2, dim=1)
#     return approx, detail

# def inverse_wavelet_transform(approx, detail, rec_filters):
#     """
#     approx, detail: [B, C, L]
#     rec_filters: [2*C, 1, K]
#     Returns:
#         x: [B, C, 2*L]
#     """
#     B, C, L = approx.shape
#     x = torch.cat([approx, detail], dim=1)  # [B, 2C, L]
#     K = rec_filters.shape[-1]
#     # Output length = stride * (L - 1) + K
#     out = F.conv_transpose1d(x, rec_filters.to(x.device), stride=2, groups=C)  # [B, C, 2L + K - 2]
#     # Crop to match exact size
#     expected_len = 2 * L
#     actual_len = out.shape[-1]
#     crop = actual_len - expected_len
#     crop_left = crop // 2
#     crop_right = crop - crop_left
#     out = out[:, :, crop_left:actual_len - crop_right]
#     return out

def create_wavelet_filter(wave: str, in_channels: int):
    w = pywt.Wavelet(wave)

    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=torch.float)  # low-pass
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=torch.float)
    base_dec_filters = torch.stack([dec_lo, dec_hi], dim=0).unsqueeze(1)  # [2, 1, K]
    dec_filters = base_dec_filters.repeat(in_channels, 1, 1)  # [2*C, 1, K]

    rec_lo = torch.tensor(w.rec_lo, dtype=torch.float)
    rec_hi = torch.tensor(w.rec_hi, dtype=torch.float)
    base_rec_filters = torch.stack([rec_lo, rec_hi], dim=0).unsqueeze(1)
    rec_filters = base_rec_filters.repeat(in_channels, 1, 1)

    return dec_filters, rec_filters

def wavelet_transform(x, dec_filters):
    B, C, L = x.shape
    K = dec_filters.shape[-1]
    pad_len = (K - 1) // 2

    front = x[:, :, 0:1].repeat(1, 1, pad_len)
    end = x[:, :, -1:].repeat(1, 1, pad_len)
    x_padded = torch.cat([front, x, end], dim=2)  # [B, C, L + 2*pad_len]

    filters = dec_filters.to(x.device)
    out = F.conv1d(x_padded, filters, stride=2, padding=0, groups=C)
    approx, detail = out.chunk(2, dim=1)
    return approx, detail

def inverse_wavelet_transform(approx, detail, rec_filters):
    B, C, L = approx.shape
    filters = rec_filters.to(approx.device)  # [2*C, 1, K]
    x = torch.cat([approx, detail], dim=1)  # [B, 2C, L]
    out = F.conv_transpose1d(x, filters, stride=2, padding=0, groups=C)  # [B, C, 2L]
    out = out[:, :, :2 * L]
    return out

<<<<<<< HEAD:models/DWDMixer.py
class DWTH(nn.Module):
    def __init__(self, configs, wavelet='haar', levels=2, seq_len=96, device='cuda'):
        super(DWTH, self).__init__()
        self.device = device
        self.levels = levels
        mode = 'symmetric' 
        self.dwt = DWT1DForward(wave=wavelet, J=1, mode=mode)  
        self.idwt = DWT1DInverse(wave=wavelet)
        self.wt_filter, self.iwt_filter = create_wavelet_filter(wavelet, configs.d_model)
        self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)
        self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)
        
        self.linear = nn.Sequential(
                    torch.nn.Linear(
                        configs.d_model,
                        configs.d_model*2,
                    ),
                    nn.GELU(),
                    torch.nn.Linear(
                        configs.d_model*2,
                        configs.d_model,
                    ),
                )
        self.convs = nn.ModuleList([
            nn.Sequential(nn.Conv1d(configs.d_model, configs.d_model, 3, padding='same', bias=True),
                           nn.Tanh()
                        ) for _ in range(configs.wt_level)
        ])
=======
>>>>>>> 8126f780ff67afdceff3b6ee134d8d200a242624:models/D2WDMixer.py


class DWTH_BLOCK(nn.Module):

    def __init__(self, configs):
        super(DWTH_BLOCK, self).__init__()
        self.DWTH_list = nn.ModuleList([
            DWTH(configs, "haar", configs.wt_level, configs.seq_len//2**i)
            for i in range(configs.down_sampling_layers+1)
            ])

        
        self.down_sampling_layers = torch.nn.ModuleList(
            [
                nn.Sequential(
                    torch.nn.Linear(
                        configs.seq_len // (2 ** i),
                        configs.seq_len // (2 ** (i+1)),
                    ),
                    nn.GELU(),
                    torch.nn.Linear(
                        configs.seq_len // (2 ** (i+1)),
                        configs.seq_len // (2 ** (i+1)),
                    ),

                )
                for i in range(configs.down_sampling_layers)
            ]
        )
       
    def forward(self, list):
        # print(list[0].shape, list[1].shape, list[2].shape, list[3].shape)
        out_list = []
        for i, x in enumerate(list):
            x = x.permute(0, 2, 1) 
            x_wave = self.DWTH_list[i](x)
            out_list.append(x_wave.permute(0, 2, 1))  
        return out_list
        # list = [tensor.permute(0, 2, 1) for tensor in list]
        # x = list[0]
        # next = list[1]
        # out_list = []
        # for i in range(len(list)):
        #     if i <= 2:
        #         x = self.DWTH_list[i](x)
        #         out_list.append(x.permute(0, 2, 1))
        #         x_res = self.down_sampling_layers[i](x)
        #         x = next + x_res
        #         if i + 2 <= len(list) - 1:
        #             next = list[i + 2]
        #     else:
        #         x = self.DWTH_list[i](x)
        #         out_list.append(x.permute(0, 2, 1))

        # return out_list

class DWTH_MIXING(nn.Module):
    def __init__(self, configs):
        super(DWTH_MIXING, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.down_sampling_window = configs.down_sampling_window

        self.layer_norm = nn.LayerNorm(configs.d_model)
        self.dropout = nn.Dropout(configs.dropout)
        self.channel_independence = configs.channel_independence

        if configs.decomp_method == 'moving_avg':
            self.decompsition = series_decomp(configs.moving_avg)
        elif configs.decomp_method == "dft_decomp":
            self.decompsition = DFT_series_decomp(configs.top_k)
        else:
            raise ValueError('decompsition is error')

        if configs.channel_independence == 0:
            self.cross_layer = nn.Sequential(
                nn.Linear(in_features=configs.d_model, out_features=configs.d_ff),
                nn.GELU(),
                nn.Linear(in_features=configs.d_ff, out_features=configs.d_model),
            )

        self.DWTH_BLOCK = DWTH_BLOCK(configs)


        self.out_cross_layer = nn.Sequential(
            nn.Linear(in_features=configs.d_model, out_features=configs.d_ff),
            nn.GELU(),
            nn.Linear(in_features=configs.d_ff, out_features=configs.d_model),
        )

    def forward(self, x_list):
        out_list = self.DWTH_BLOCK(x_list)
        return out_list
        # out_list = []
        #     for ori, out_season, out_trend, length in zip(x_list, out_season_list, out_trend_list,
        #                                                 length_list):
        #         out = out_season + out_trend
        #         if self.channel_independence:
        #             out = ori + self.out_cross_layer(out)
        #         out_list.append(out[:, :length, :])
        #     return out_list


class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.down_sampling_window = configs.down_sampling_window
        self.channel_independence = configs.channel_independence
        self.DWTH_blocks = nn.ModuleList([DWTH_MIXING(configs)
                                         for _ in range(configs.e_layers)])

        self.preprocess = series_decomp(configs.moving_avg)
        self.enc_in = configs.enc_in
        self.use_future_temporal_feature = configs.use_future_temporal_feature
        self.predict_linear = torch.nn.ModuleList(
            [
                nn.Sequential(
                    torch.nn.Linear(
                        configs.seq_len // (2 ** i),
                        configs.seq_len,
                    ))
                for i in range(configs.down_sampling_layers+1)
            ]
        )
        if self.channel_independence == 1:
            self.enc_embedding = DataEmbedding_wo_pos(1, configs.d_model, configs.embed, configs.freq,
                                                      configs.dropout)
        else:
            self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                      configs.dropout)

        self.layer = configs.e_layers

        self.normalize_layers = torch.nn.ModuleList(
            [
                Normalize(self.configs.enc_in, affine=True, non_norm=True if configs.use_norm == 0 else False)
                for i in range(configs.down_sampling_layers + 1)
            ]
        )

        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.predict_layers = torch.nn.ModuleList(
                [
                    torch.nn.Linear(
                        configs.seq_len,
                        configs.pred_len,
                    )
                    for i in range(configs.down_sampling_layers + 1)
                ]
            )

            if self.channel_independence == 1:
                self.projection_layer = nn.Linear(
                    configs.d_model, 1, bias=True)
                self.out_res_layers = torch.nn.ModuleList([
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** i),
                        configs.seq_len // (configs.down_sampling_window ** i),
                    )
                    for i in range(configs.down_sampling_layers + 1)
                ])

                self.regression_layers = torch.nn.ModuleList(
                    [
                        torch.nn.Linear(
                            configs.seq_len // (configs.down_sampling_window ** i),
                            configs.pred_len,
                        )
                        for i in range(configs.down_sampling_layers + 1)
                    ]
                )
            else:
                self.projection_layer = nn.Linear(
                    configs.d_model, configs.c_out, bias=True)

                self.out_res_layers = torch.nn.ModuleList([
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** i),
                        configs.seq_len // (configs.down_sampling_window ** i),
                    )
                    for i in range(configs.down_sampling_layers + 1)
                ])

                self.regression_layers = torch.nn.ModuleList(
                    [
                        torch.nn.Linear(
                            configs.seq_len // (configs.down_sampling_window ** i),
                            configs.pred_len,
                        )
                        for i in range(configs.down_sampling_layers + 1)
                    ]
                )
        if self.task_name == 'imputation' or self.task_name == 'anomaly_detection':
            if self.channel_independence == 1:
                self.projection_layer = nn.Linear(
                    configs.d_model, 1, bias=True)
            else:
                self.projection_layer = nn.Linear(
                    configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                configs.d_model * configs.seq_len, configs.num_class)

    def out_projection(self, dec_out, i, out_res):
        dec_out = self.projection_layer(dec_out)
        out_res = out_res.permute(0, 2, 1)
        out_res = self.out_res_layers[i](out_res)
        out_res = self.regression_layers[i](out_res).permute(0, 2, 1)
        dec_out = dec_out + out_res
        return dec_out

    def SeasonalTrendDecomposition(self, x_list):
            out1_list = []
            out2_list = []
            for x in x_list:
                x_1, x_2 = self.preprocess(x)
                out1_list.append(x_1)
                out2_list.append(x_2)
            return (out1_list, out2_list)

    def __multi_scale_process_inputs(self, x_enc, x_mark_enc):
        if self.configs.down_sampling_method == 'max':
            down_pool = torch.nn.MaxPool1d(self.configs.down_sampling_window, return_indices=False)
        elif self.configs.down_sampling_method == 'avg':
            down_pool = torch.nn.AvgPool1d(self.configs.down_sampling_window)
        elif self.configs.down_sampling_method == 'conv':
            padding = 1 if torch.__version__ >= '1.5.0' else 2
            down_pool = nn.Conv1d(in_channels=self.configs.enc_in, out_channels=self.configs.enc_in,
                                  kernel_size=3, padding=padding,
                                  stride=self.configs.down_sampling_window,
                                  padding_mode='circular',
                                  bias=False)
        else:
            return x_enc, x_mark_enc
        # B,T,C -> B,C,T
        x_enc = x_enc.permute(0, 2, 1)

        x_enc_ori = x_enc
        x_mark_enc_mark_ori = x_mark_enc

        x_enc_sampling_list = []
        x_mark_sampling_list = []
        x_enc_sampling_list.append(x_enc.permute(0, 2, 1))
        x_mark_sampling_list.append(x_mark_enc)

        for i in range(self.configs.down_sampling_layers):
            x_enc_sampling = down_pool(x_enc_ori)

            x_enc_sampling_list.append(x_enc_sampling.permute(0, 2, 1))
            x_enc_ori = x_enc_sampling

            if x_mark_enc_mark_ori is not None:
                x_mark_sampling_list.append(x_mark_enc_mark_ori[:, ::self.configs.down_sampling_window, :])
                x_mark_enc_mark_ori = x_mark_enc_mark_ori[:, ::self.configs.down_sampling_window, :]

        x_enc = x_enc_sampling_list
        if x_mark_enc_mark_ori is not None:
            x_mark_enc = x_mark_sampling_list
        else:
            x_mark_enc = x_mark_enc

        return x_enc, x_mark_enc

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        
        if self.use_future_temporal_feature:
            if self.channel_independence == 1:
                B, T, N = x_enc.size()
                x_mark_dec = x_mark_dec.repeat(N, 1, 1)
                self.x_mark_dec = self.enc_embedding(None, x_mark_dec)
            else:
                self.x_mark_dec = self.enc_embedding(None, x_mark_dec)

        x_enc, x_mark_enc = self.__multi_scale_process_inputs(x_enc, x_mark_enc)

        x_list = []
        x_mark_list = []
        if x_mark_enc is not None:
            for i, x, x_mark in zip(range(len(x_enc)), x_enc, x_mark_enc):
                B, T, N = x.size()
                x = self.normalize_layers[i](x, 'norm')
                if self.channel_independence == 1:
                    x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
                    x_mark = x_mark.repeat(N, 1, 1)
                x_list.append(x)
                x_mark_list.append(x_mark)
        else:
            for i, x in zip(range(len(x_enc)), x_enc, ):
                B, T, N = x.size()
                x = self.normalize_layers[i](x, 'norm')
                if self.channel_independence == 1:
                    x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
                x_list.append(x)

        enc_out_list = []
        x_list = self.SeasonalTrendDecomposition(x_list)
        if x_mark_enc is not None:
            for i, x, x_mark in zip(range(len(x_list[0])), x_list[0], x_mark_list):
                enc_out = self.enc_embedding(x, x_mark)  # [B,T,C]
                enc_out = self.predict_linear[i](enc_out.permute(0, 2, 1)).permute( 0, 2, 1)                
                enc_out_list.append(enc_out)
        else:
            for i, x in zip(range(len(x_list[0])), x_list[0]):
                enc_out = self.enc_embedding(x, None)  # [B,T,C]
                enc_out = self.predict_linear[i](enc_out.permute(0, 2, 1)).permute( 0, 2, 1)                
                enc_out_list.append(enc_out)
        # enc_out_list = enc_out_list[::-1]
        for i in range(self.layer):
            enc_out_list = self.DWTH_blocks[i](enc_out_list)
        # enc_out_list = enc_out_list[::-1]
        dec_out_list = self.future_multi_mixing(B, enc_out_list, x_list)

        dec_out = torch.stack(dec_out_list, dim=-1).sum(-1)
        dec_out = self.normalize_layers[0](dec_out, 'denorm')
        return dec_out

    def future_multi_mixing(self, B, enc_out_list, x_list):
        dec_out_list = []
        if self.channel_independence == 1:
            for i, enc_out, out_res in zip(range(len(x_list[0])), enc_out_list, x_list[1]):
                dec_out = self.predict_layers[i](enc_out.permute(0, 2, 1)).permute(
                    0, 2, 1)  # align temporal dimension
                dec_out = self.out_projection(dec_out, i, out_res)
                dec_out = dec_out.reshape(B, self.configs.c_out, self.pred_len).permute(0, 2, 1).contiguous()
                dec_out_list.append(dec_out)

        else:
            for i, enc_out, out_res in zip(range(len(x_list[0])), enc_out_list, x_list[1]):
                dec_out = self.predict_layers[i](enc_out.permute(0, 2, 1)).permute(
                    0, 2, 1)  # align temporal dimension
                dec_out = self.out_projection(dec_out, i, out_res)
                dec_out_list.append(dec_out)

        return dec_out_list

    def classification(self, x_enc, x_mark_enc):
        x_enc, _ = self.__multi_scale_process_inputs(x_enc, None)
        x_list = x_enc

        # embedding
        enc_out_list = []
        for x in x_list:
            enc_out = self.enc_embedding(x, None)  # [B,T,C]
            enc_out_list.append(enc_out)

        # MultiScale-CrissCrossAttention  as encoder for past
        for i in range(self.layer):
            enc_out_list = self.DWTH_blocks[i](enc_out_list)

        enc_out = enc_out_list[0]
        # Output
        # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.act(enc_out)
        output = self.dropout(output)
        # zero-out padding embeddings
        output = output * x_mark_enc.unsqueeze(-1)
        # (batch_size, seq_length * d_model)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def anomaly_detection(self, x_enc):
        B, T, N = x_enc.size()
        x_enc, _ = self.__multi_scale_process_inputs(x_enc, None)

        x_list = []

        for i, x in zip(range(len(x_enc)), x_enc, ):
            B, T, N = x.size()
            x = self.normalize_layers[i](x, 'norm')
            if self.channel_independence == 1:
                x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
            x_list.append(x)

        # embedding
        enc_out_list = []
        for x in x_list:
            enc_out = self.enc_embedding(x, None)  # [B,T,C]
            enc_out_list.append(enc_out)

        # MultiScale-CrissCrossAttention  as encoder for past
        for i in range(self.layer):
            enc_out_list = self.DWTH_blocks[i](enc_out_list)

        dec_out = self.projection_layer(enc_out_list[0])
        dec_out = dec_out.reshape(B, self.configs.c_out, -1).permute(0, 2, 1).contiguous()

        dec_out = self.normalize_layers[0](dec_out, 'denorm')
        return dec_out

    def imputation(self, x_enc, x_mark_enc, mask):
        means = torch.sum(x_enc, dim=1) / torch.sum(mask == 1, dim=1)
        means = means.unsqueeze(1).detach()
        x_enc = x_enc - means
        x_enc = x_enc.masked_fill(mask == 0, 0)
        stdev = torch.sqrt(torch.sum(x_enc * x_enc, dim=1) /
                           torch.sum(mask == 1, dim=1) + 1e-5)
        stdev = stdev.unsqueeze(1).detach()
        x_enc /= stdev

        B, T, N = x_enc.size()
        x_enc, x_mark_enc = self.__multi_scale_process_inputs(x_enc, x_mark_enc)

        x_list = []
        x_mark_list = []
        if x_mark_enc is not None:
            for i, x, x_mark in zip(range(len(x_enc)), x_enc, x_mark_enc):
                B, T, N = x.size()
                if self.channel_independence == 1:
                    x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
                x_list.append(x)
                x_mark = x_mark.repeat(N, 1, 1)
                x_mark_list.append(x_mark)
        else:
            for i, x in zip(range(len(x_enc)), x_enc, ):
                B, T, N = x.size()
                if self.channel_independence == 1:
                    x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
                x_list.append(x)

        # embedding
        enc_out_list = []
        for x in x_list:
            enc_out = self.enc_embedding(x, None)  # [B,T,C]
            enc_out_list.append(enc_out)

        # MultiScale-CrissCrossAttention  as encoder for past
        for i in range(self.layer):
            enc_out_list = self.DWTH_blocks[i](enc_out_list)

        dec_out = self.projection_layer(enc_out_list[0])
        dec_out = dec_out.reshape(B, self.configs.c_out, -1).permute(0, 2, 1).contiguous()

        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc, x_mark_enc, mask)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        else:
            raise ValueError('Other tasks implemented yet')
