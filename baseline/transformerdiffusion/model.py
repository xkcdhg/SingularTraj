import math
import torch
import torch.nn as nn
from torch.nn import Module, Linear
import numpy as np
from .layers import PositionalEncoding, ConcatSquashLinear


class st_encoder(nn.Module):
    """Transformer Denoising Model
    codebase borrowed from https://github.com/MediaBrain-SJTU/LED"""
    def __init__(self):
        super().__init__()
        channel_in = 2
        channel_out = 32
        dim_kernel = 3
        self.dim_embedding_key = 256
        self.spatial_conv = nn.Conv1d(channel_in, channel_out, dim_kernel, stride=1, padding=1)
        self.temporal_encoder = nn.GRU(channel_out, self.dim_embedding_key, 1, batch_first=True)
        self.relu = nn.ReLU()
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.spatial_conv.weight)
        nn.init.kaiming_normal_(self.temporal_encoder.weight_ih_l0)
        nn.init.kaiming_normal_(self.temporal_encoder.weight_hh_l0)
        nn.init.zeros_(self.spatial_conv.bias)
        nn.init.zeros_(self.temporal_encoder.bias_ih_l0)
        nn.init.zeros_(self.temporal_encoder.bias_hh_l0)

    def forward(self, X):
        X_t = torch.transpose(X, 1, 2)
        X_after_spatial = self.relu(self.spatial_conv(X_t))
        X_embed = torch.transpose(X_after_spatial, 1, 2)
        output_x, state_x = self.temporal_encoder(X_embed)
        state_x = state_x.squeeze(0)
        return state_x


class social_transformer(nn.Module):
    """Transformer Denoising Model
    codebase borrowed from https://github.com/MediaBrain-SJTU/LED"""
    def __init__(self, cfg):
        super(social_transformer, self).__init__()
        self.encode_past = nn.Linear(cfg.k*cfg.s+6, 256, bias=False)
        self.layer = nn.TransformerEncoderLayer(d_model=256, nhead=2, dim_feedforward=256)
        self.transformer_encoder = nn.TransformerEncoder(self.layer, num_layers=2)

    def forward(self, h, mask):
        h_feat = self.encode_past(h.reshape(h.size(0), -1)).unsqueeze(1)
        h_feat_ = self.transformer_encoder(h_feat, mask)
        h_feat = h_feat + h_feat_

        return h_feat


class TransformerDenoisingModel(Module):
    """Transformer Denoising Model
    codebase borrowed from https://github.com/MediaBrain-SJTU/LED"""
    def __init__(self, context_dim=256, cfg=None):
        super().__init__()
        self.context_dim = context_dim
        self.spatial_dim = 1
        self.temporal_dim = cfg.k
        self.n_samples = cfg.s
        self.encoder_context = social_transformer(cfg)
        self.pos_emb = PositionalEncoding(d_model=2*context_dim, dropout=0.1, max_len=24)
        self.concat1 = ConcatSquashLinear(self.n_samples*self.spatial_dim*self.temporal_dim, 2*context_dim, context_dim+3)
        self.concat3 = ConcatSquashLinear(2*context_dim,context_dim,context_dim+3)
        self.concat4 = ConcatSquashLinear(context_dim,context_dim//2,context_dim+3)
        self.linear = ConcatSquashLinear(context_dim//2, self.n_samples*self.spatial_dim*self.temporal_dim, context_dim+3)

    def forward(self, x, beta, context, mask):
        batch_size = x.size(0)
        beta = beta.view(batch_size, 1, 1)
        time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)
        ctx_emb = torch.cat([time_emb, context], dim=-1)
        x = self.concat1(ctx_emb, x)
        final_emb = x.permute(1,0,2)
        final_emb = self.pos_emb(final_emb)
        trans = self.transformer_encoder(final_emb).permute(1,0,2)
        trans = self.concat3(ctx_emb, trans)
        trans = self.concat4(ctx_emb, trans)
        return self.linear(ctx_emb, trans)

    def encode_context(self, context, mask):
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        context = self.encoder_context(context, mask)
        return context

    def generate_accelerate(self, x, beta, context, mask):
        beta = beta.view(beta.size(0), 1)
        time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1) 
        ctx_emb = torch.cat([time_emb, context.view(-1, self.context_dim*self.spatial_dim)], dim=-1)

        trans = self.concat1.batch_generate(ctx_emb, x.view(-1, self.n_samples*self.temporal_dim*self.spatial_dim))
        trans = self.concat3.batch_generate(ctx_emb, trans)  
        trans = self.concat4.batch_generate(ctx_emb, trans)
        return self.linear.batch_generate(ctx_emb, trans).view(-1, self.n_samples, self.temporal_dim, self.spatial_dim)


class DiffusionModel(Module):
    """DiffusionModel with RK4 ODE solver (DiffTrajectory Eq. 9-11). No ADSS."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = TransformerDenoisingModel(context_dim=256, cfg=cfg)

        self.betas = self.make_beta_schedule(
            schedule=self.cfg.beta_schedule,
            n_timesteps=self.cfg.steps,
            start=self.cfg.beta_start,
            end=self.cfg.beta_end,
        ).cuda()

        self.alphas = 1 - self.betas
        self.alphas_prod = torch.cumprod(self.alphas, 0)
        self.alphas_bar_sqrt = torch.sqrt(self.alphas_prod)
        self.one_minus_alphas_bar_sqrt = torch.sqrt(1 - self.alphas_prod)
        self.reset_sampling_stats()

    def make_beta_schedule(self, schedule='linear', n_timesteps=1000,
                           start=1e-5, end=1e-2):
        if schedule == 'linear':
            betas = torch.linspace(start, end, n_timesteps)
        elif schedule == 'quad':
            betas = torch.linspace(start**0.5, end**0.5, n_timesteps) ** 2
        elif schedule == 'sigmoid':
            betas = torch.linspace(-6, 6, n_timesteps)
            betas = torch.sigmoid(betas) * (end - start) + start
        return betas
    
    def reset_sampling_stats(self):
        self.sampling_stats = {
            "nfe": 0,
            "nan_batches": 0,
            "inf_batches": 0,
            "max_abs_value": 0.0,
        }

    def extract(self, input, t, x):
        shape = x.shape
        out = torch.gather(input, 0, t.to(input.device))
        reshape = [t.shape[0]] + [1] * (len(shape) - 1)
        return out.reshape(*reshape)

    def forward(self, past_traj, traj_mask, loc):
        return self.p_sample_loop_rk4(past_traj, traj_mask, loc)

    def _F(self, x, beta, sigma, context, mask):
        """ODE drift — DiffTrajectory Eq. 9.
        Accepts pre-extracted beta/sigma to avoid redundant gather ops.
        """
        eps_theta = self.model.generate_accelerate(x, beta, context, mask)
        f_term     = -0.5 * beta * x
        score_term = -0.5 * beta * (eps_theta / (sigma + 1e-4))
        self.sampling_stats["nfe"] += 1

        if torch.isnan(x).any() or torch.isnan(eps_theta).any():
            self.sampling_stats["nan_batches"] += 1

        if torch.isinf(x).any() or torch.isinf(eps_theta).any():
            self.sampling_stats["inf_batches"] += 1
        self.sampling_stats["max_abs_value"] = max(
            self.sampling_stats["max_abs_value"],
            x.abs().max().item(),
            eps_theta.abs().max().item(),
        )
        return f_term + score_term

    def _get_beta_sigma(self, t_int, batch_size, device, ref):
        """Extract beta and sigma for a given timestep — called once per
        unique t value rather than inside every _F call."""
        t_idx = max(0, min(t_int, self.cfg.steps - 1))
        t_batch = torch.full((batch_size,), t_idx, dtype=torch.long, device=device)
        beta  = self.extract(self.betas, t_batch, ref)
        sigma = self.extract(self.one_minus_alphas_bar_sqrt, t_batch, ref)
        return beta, sigma

    def _rk4_step(self, x, t, delta, context, mask):
        """One RK4 step — DiffTrajectory Eq. 10-11."""
        t_mid  = max(0, int(round(t - delta / 2)))
        t_next = max(0, int(round(t - delta)))
        B, device = x.shape[0], x.device

        # Pre-extract beta/sigma for each unique timestep (3 unique values)
        beta_t,    sigma_t    = self._get_beta_sigma(t,      B, device, x)
        beta_mid,  sigma_mid  = self._get_beta_sigma(t_mid,  B, device, x)
        beta_next, sigma_next = self._get_beta_sigma(t_next, B, device, x)

        k1 = delta * self._F(x,          beta_t,    sigma_t,    context, mask)
        k2 = delta * self._F(x + k1/2,   beta_mid,  sigma_mid,  context, mask)
        k3 = delta * self._F(x + k2/2,   beta_mid,  sigma_mid,  context, mask)
        k4 = delta * self._F(x + k3,     beta_next, sigma_next, context, mask)

        return x - (1.0 / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)

    def p_sample_loop_rk4(self, x, mask, loc):
        self.reset_sampling_stats()
        rng = np.random.default_rng(seed=0)
        cur_y = torch.tensor(
            rng.normal(loc=0, scale=1.0, size=loc.shape),
            dtype=torch.float32, device=x.device,
        )

        # encode_context handles -inf fill internally — pass raw mask
        context = self.model.encode_context(x, mask)

        # Pre-process mask once for generate_accelerate
        mask_f = (mask.float()
                  .masked_fill(mask == 0, float('-inf'))
                  .masked_fill(mask == 1, float(0.0)))

        for t in reversed(range(self.cfg.steps)):
            cur_y = self._rk4_step(cur_y, t, delta=1.0, context=context, mask=mask_f)

        return cur_y
    