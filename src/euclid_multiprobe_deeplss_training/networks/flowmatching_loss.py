import math
import torch
import torch.nn as nn
from torchdiffeq import odeint
from typing import Callable, Optional, Literal



class SinusoidalTimeEmbedding(nn.Module):
    """
    Sinusoidal embedding for scalar flow time.

    Input:
        t: [B, 1] or [B]

    Output:
        emb: [B, dim]
    """

    def __init__(self, dim: int, max_period: float = 10_000.0):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: [B, 1] or [B]
        if t.ndim == 2:
            t = t.squeeze(-1)

        # t: [B]
        half_dim = self.dim // 2

        # freqs: [half_dim]
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(
                half_dim,
                device=t.device,
                dtype=t.dtype,
            )
            / max(half_dim - 1, 1)
        )

        # args: [B, half_dim]
        args = t[:, None] * freqs[None, :]

        # emb: [B, 2 * half_dim]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

        if emb.shape[-1] < self.dim:
            # emb: [B, dim]
            emb = F.pad(emb, (0, self.dim - emb.shape[-1]))

        return emb

class StandardTransformerBlock(nn.Module):
    """
    Standard pre-norm transformer block.

    Input:
        x: [B, S, D]

    Output:
        x: [B, S, D]

    where:
        B = batch size
        S = sequence length
        D = transformer width
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.norm_attn = nn.LayerNorm(d_model)

        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm_mlp = nn.LayerNorm(d_model)

        hidden_dim = int(mlp_ratio * d_model)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # x: [B, S, D]

        # Pre-norm self-attention.
        x_norm = self.norm_attn(x)  # [B, S, D]

        attn_out, _ = self.attn(
            query=x_norm,                  # [B, S, D]
            key=x_norm,                    # [B, S, D]
            value=x_norm,                  # [B, S, D]
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )

        # Residual connection.
        x = x + self.dropout(attn_out)     # [B, S, D]

        # Pre-norm MLP.
        mlp_out = self.mlp(self.norm_mlp(x))  # [B, S, D]

        # Residual connection.
        x = x + mlp_out                    # [B, S, D]

        return x


class PosteriorVectorFieldTransformer(nn.Module):
    """
    Transformer posterior vector field using one image-conditioning token.

    The sequence is:

        [image_token, y_0_token, y_1_token, ..., y_{M-1}_token]

    Inputs:
        u_t: [B, M]
        t:   [B, 1] or [B]
        h:   [B, H_img]

    Output:
        pred_v: [B, M]
    """

    def __init__(
        self,
        label_dims: int,
        embedding_dim: int,
        d_model: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model={d_model} must be divisible by num_heads={num_heads}."
            )

        self.label_dims = label_dims
        self.embedding_dim = embedding_dim
        self.d_model = d_model

        # Project one scalar u_t[j] into a token.
        #
        # Input:
        #     u_t[..., None]: [B, M, 1]
        #
        # Output:
        #     label_tokens: [B, M, D]
        self.scalar_to_token = nn.Linear(1, d_model)

        # Learned coordinate embeddings for y[0], ..., y[M-1].
        #
        # coord_embedding: [M, D]
        self.coord_embedding = nn.Embedding(label_dims, d_model)

        # Project image embedding h into one transformer token.
        #
        # h:           [B, H_img]
        # image_token: [B, D]
        self.image_to_token = nn.Linear(embedding_dim, d_model)

        # Time embedding.
        #
        # t:        [B, 1]
        # time_emb: [B, D]
        self.time_embedding = SinusoidalTimeEmbedding(d_model)

        self.time_mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.SiLU(),
            nn.Linear(4 * d_model, d_model),
        )

        self.blocks = nn.ModuleList(
            [
                StandardTransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.final_norm = nn.LayerNorm(d_model)

        # Convert each final label token into one scalar velocity.
        #
        # Input:
        #     final_label_tokens: [B, M, D]
        #
        # Output:
        #     pred_v[..., None]: [B, M, 1]
        self.token_to_velocity = nn.Linear(d_model, 1)

    def forward(
        self,
        u_t: torch.Tensor,
        t: torch.Tensor,
        h: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        u_t:
            Current label-space state.
            Shape: [B, M]

        t:
            Flow time.
            Shape: [B, 1] or [B]

        h:
            Single image embedding from image_encoder(X).
            Shape: [B, H_img]

        Returns
        -------
        pred_v:
            Predicted velocity.
            Shape: [B, M]
        """

        # u_t: [B, M]
        if u_t.ndim != 2:
            raise ValueError(f"u_t must have shape [B, M], got {u_t.shape}.")

        B, M = u_t.shape

        if M != self.label_dims:
            raise ValueError(
                f"Expected M={self.label_dims}, got M={M}."
            )

        # h: [B, H_img]
        if h.ndim != 2:
            raise ValueError(
                f"h must have shape [B, H_img] for this single-token model, got {h.shape}."
            )

        if h.shape[0] != B:
            raise ValueError(
                f"h batch size {h.shape[0]} does not match u_t batch size {B}."
            )

        if t.ndim == 1:
            # t: [B] -> [B, 1]
            t = t[:, None]

        if t.shape[0] != B:
            raise ValueError(
                f"t batch size {t.shape[0]} does not match u_t batch size {B}."
            )

        # ------------------------------------------------------------
        # Build label tokens.
        # ------------------------------------------------------------

        # u_t_scalar: [B, M, 1]
        u_t_scalar = u_t.unsqueeze(-1)

        # label_tokens: [B, M, D]
        label_tokens = self.scalar_to_token(u_t_scalar)

        # coord_ids: [M]
        coord_ids = torch.arange(M, device=u_t.device)

        # coord_emb: [M, D]
        coord_emb = self.coord_embedding(coord_ids)

        # label_tokens: [B, M, D]
        label_tokens = label_tokens + coord_emb[None, :, :]

        # ------------------------------------------------------------
        # Add time embedding to label tokens.
        # ------------------------------------------------------------

        # t: [B, 1]
        t = t.to(device=u_t.device, dtype=u_t.dtype)

        # time_emb: [B, D]
        time_emb = self.time_embedding(t)

        # time_emb: [B, D]
        time_emb = self.time_mlp(time_emb)

        # label_tokens: [B, M, D]
        label_tokens = label_tokens + time_emb[:, None, :]

        # ------------------------------------------------------------
        # Build one image-conditioning token.
        # ------------------------------------------------------------

        # h: [B, H_img]
        h = h.to(device=u_t.device, dtype=u_t.dtype)

        # image_token: [B, D]
        image_token = self.image_to_token(h)

        # image_token: [B, 1, D]
        image_token = image_token[:, None, :]

        # ------------------------------------------------------------
        # Concatenate image token and label tokens.
        # ------------------------------------------------------------

        # seq: [B, 1 + M, D]
        #
        # seq[:, 0, :]      is the image-conditioning token.
        # seq[:, 1:, :]     are the M label tokens.
        seq = torch.cat([image_token, label_tokens], dim=1)

        # ------------------------------------------------------------
        # Transformer blocks.
        # ------------------------------------------------------------

        for block in self.blocks:
            # seq: [B, 1 + M, D]
            seq = block(seq)

        # ------------------------------------------------------------
        # Read out only label tokens.
        # ------------------------------------------------------------

        # label_out: [B, M, D]
        label_out = seq[:, 1:, :]

        # label_out: [B, M, D]
        label_out = self.final_norm(label_out)

        # pred_v: [B, M, 1]
        pred_v = self.token_to_velocity(label_out)

        # pred_v: [B, M]
        pred_v = pred_v.squeeze(-1)

        return pred_v





class PosteriorODEFunc(nn.Module):
    """
    ODE function wrapper for torchdiffeq.

    Converts torchdiffeq's scalar time t into the fixed-shape
    time tensor expected by PosteriorVectorFieldTransformer.

    Input from torchdiffeq:
        scalar_t: []
        u:        [B, M]

    Internal model call:
        posterior_vector_field(
            u_t=u,        # [B, M]
            t=t_batch,    # [B, 1]
            h=h           # [B, H_img]
        )

    Output:
        du_dt: [B, M]
    """

    def __init__(
        self,
        posterior_vector_field: nn.Module,
        h: torch.Tensor,
    ):
        super().__init__()

        self.posterior_vector_field = posterior_vector_field
        self.h = h

        # Fixed dimensions from the trained vector field.
        self.y_dim = posterior_vector_field.label_dims            # M

        # h: [B, H_img]
        assert h.ndim == 2
        assert h.shape[1] == posterior_vector_field.embedding_dim

    def forward(
        self,
        scalar_t: torch.Tensor,
        u: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        scalar_t:
            Scalar ODE time from torchdiffeq.
            Shape: []

        u:
            Current label-space state.
            Shape: [B, M]

        Returns
        -------
        du_dt:
            Velocity.
            Shape: [B, M]
        """

        # u: [B, M]
        assert u.ndim == 2
        
        # scalar_t: []
        assert scalar_t.ndim == 0

        # t_batch: [B, 1]
        batch_size = u.shape[0]
        t_batch = scalar_t.expand(batch_size, 1)

        assert t_batch.shape == (batch_size, 1)
        assert t_batch.dtype == u.dtype
        assert t_batch.device == u.device

        # du_dt: [B, M]
        du_dt = self.posterior_vector_field(
            u_t=u,
            t=t_batch,
            h=self.h,
        )

        assert du_dt.shape == (batch_size, self.y_dim)

        return du_dt


class CNFFMModel(nn.Module):
    """
    Posterior flow matching model.
    """

    def __init__(self, encoder, y_dim, vectorfield_kwargs):
        super().__init__()

        self.y_dim = y_dim
        self.encoder = encoder
        self.vector_field = PosteriorVectorFieldTransformer(label_dims=y_dim, embedding_dim=encoder.embed_dim, **vectorfield_kwargs)
        self.loss = nn.MSELoss()

    def forward(self, X, y):
        """
        X: input image data [B, N, C]
        y: target vector labels [B, M]
        """

        #
        # check inputs
        #

        # X: [B, N, C]
        assert X.ndim == 3
        batch_size = X.shape[0]

        # y: [B, M]
        assert y.ndim == 2
        assert y.shape[0] == batch_size


        # u1: [B, M]
        # Assumes y is already normalized and unconstrained.
        u1 = y

        #
        # encode image
        #

        # h: [B, H_img] 
        h = self.encoder(X)

        #
        # generate t0 variables
        # 
        
        u0 = torch.randn_like(u1)
        
        # t: [B, 1]
        t = torch.rand(batch_size, 1, device=u1.device, dtype=u1.dtype)

        # Avoid exact t = 0 or t = 1.
        eps = 1e-4
        t = eps + (1.0 - 2.0 * eps) * t

        #
        # Build interpolated state
        #

        # ut: [B, M]
        # Linear interpolation between base noise u0 and data label u1.
        ut = (1.0 - t) * u0 + t * u1


        # Compute target velocity
        
        # target_v: [B, M]
        # For the linear path, du_t / dt = u1 - u0.
        target_v = u1 - u0


        # pred_v: [B, M]
        pred_v = self.vector_field(u_t=ut, t=t, h=h)

        #
        # compute loss
        #

        # loss: scalar
        loss = self.loss(pred_v, target_v)

        return loss

    @torch.no_grad()
    def predict(self, X, num_samples: int = 4, method: Literal["dopri5", "rk4", "euler", "midpoint"] = "dopri5", rtol: float = 1e-5, atol: float = 1e-5):
        """
        Sample y ~ q_phi(y | X) using torchdiffeq.odeint.

        Assumed fixed shapes:
            X_batch: [B, C, N, N]
            h:       [B, H_img]
            u0:      [B, M]
            u1:      [B, M]
            y:       [B, M]

        Returns
        -------
        y_batch: [B, M]

        """

        self.encoder.eval()
        self.vector_field.eval()
        batch_size = X.shape[0]

        #
        # check inputs
        #

        # X: [B, N, C]
        assert X.ndim == 3
        
        
        # 
        # 1. Encode X once.
        # 

        # h: [B, H_img]
        h = self.encoder(X)

        # 
        # 2. Build ODE function du/dt = v_phi(u, t, h).
        # 

        ode_func = PosteriorODEFunc(
            posterior_vector_field=self.vector_field,
            h=h,
        )

        # t_span: [2]
        # Integrate from t = 0 to t = 1.
        t_span = torch.tensor(
            [0.0, 1.0],
            device=X.device,
            dtype=X.dtype,
        )

        all_samples = []
        for _ in range(num_samples):

            # 
            # 3. Sample initial base noise.
            # 

            # u0: [B, M]
            u0 = torch.randn(
                batch_size,
                self.y_dim,
                device=X.device,
                dtype=X.dtype,
            )

            # 
            # 4. Solve ODE.
            # 
            #
            # trajectory: [2, B, M]
            #
            # trajectory[0] is approximately u(t=0)
            # trajectory[1] is approximately u(t=1)

            trajectory = odeint(
                func=ode_func,
                y0=u0,
                t=t_span,
                method=method,
                rtol=rtol,
                atol=atol,
            )

            assert trajectory.shape == (2, batch_size, self.y_dim)

            # u1: [B, M]
            u1 = trajectory[-1]

            assert u1.shape == (batch_size, self.y_dim)

            all_samples.append(u1)

        all_samples = torch.stack(all_samples, dim=0)

        y_sample = torch.mean(all_samples, dim=0)

        # [B, M]
        return y_sample




























# class SinusoidalTimeEmbedding(nn.Module):
#     """
#     Fixed-shape sinusoidal embedding for scalar flow time t.

#     Expected input:
#         t: [B, 1]

#     Output:
#         emb: [B, time_dim]
#     """

#     def __init__(
#         self,
#         batch_size: int,
#         time_dim: int,
#         dtype: torch.dtype = torch.float32,
#         device: torch.device | str | None = None,
#     ):
#         super().__init__()

#         assert batch_size > 0
#         assert time_dim > 0
#         assert time_dim % 2 == 0

#         self.batch_size = batch_size        # B
#         self.time_dim = time_dim            # time_dim

#         half_dim = time_dim // 2            # time_dim / 2

#         # frequencies: [time_dim // 2]
#         frequencies = torch.exp(
#             -math.log(10000.0)
#             * torch.arange(half_dim, dtype=dtype, device=device)
#             / max(half_dim - 1, 1)
#         )

#         # Stored as a buffer so it moves with model.to(device/dtype), but is
#         # not a trainable parameter.
#         self.register_buffer("frequencies", frequencies, persistent=False)

#     def forward(self, t: torch.Tensor) -> torch.Tensor:
#         """
#         Parameters
#         ----------
#         t:
#             Flow time.
#             Shape: [B, 1]

#         Returns
#         -------
#         emb:
#             Sinusoidal time embedding.
#             Shape: [B, time_dim]
#         """

#         # t: [B, 1]
#         assert t.ndim == 2
#         assert t.shape == (self.batch_size, 1)

#         # Enforce dtype/device consistency with this module's buffer.
#         assert t.dtype == self.frequencies.dtype
#         assert t.device == self.frequencies.device

#         # frequencies: [time_dim // 2]
#         assert self.frequencies.shape == (self.time_dim // 2,)

#         # args: [B, time_dim // 2]
#         args = 2.0 * math.pi * t * self.frequencies[None, :]

#         # sin_part: [B, time_dim // 2]
#         sin_part = torch.sin(args)

#         # cos_part: [B, time_dim // 2]
#         cos_part = torch.cos(args)

#         # emb: [B, time_dim]
#         emb = torch.cat([sin_part, cos_part], dim=-1)

#         assert emb.shape == (self.batch_size, self.time_dim)

#         return emb


# class PosteriorVectorFieldTransformer(nn.Module):
#     """
#     Fixed-shape Transformer posterior vector field.

#     This implements:

#         pred_v = v_phi(u_t, t, h)

#     for posterior label flow matching:

#         q_phi(y | X)

#     Expected inputs:
#         u_t: [B, M]
#         t:   [B, 1]
#         h:   [B, H_img]

#     Output:
#         pred_v: [B, M]

#     Shape meanings:
#         B      = fixed batch size
#         M      = label dimension
#         H      = image encoder output dimension
#         D      = Transformer hidden dimension
#         K      = number of learned condition-memory tokens
#     """

#     def __init__(
#         self,
#         batch_size: int,
#         y_dim: int,
#         h_dim: int = 128,
#         d_model: int = 128,
#         n_heads: int = 4,
#         n_layers: int = 3,
#         dim_feedforward: int = 512,
#         time_dim: int = 64,
#         num_condition_tokens: int = 4,
#         dropout: float = 0.0,
#         dtype: torch.dtype = torch.float32,
#         device: torch.device | str | None = None,
#     ):
#         super().__init__()

#         assert batch_size > 0
#         assert y_dim > 0
#         assert h_dim > 0
#         assert d_model > 0
#         assert n_heads > 0
#         assert d_model % n_heads == 0
#         assert n_layers > 0
#         assert dim_feedforward > 0
#         assert time_dim > 0
#         assert time_dim % 2 == 0
#         assert num_condition_tokens > 0

#         self.batch_size = batch_size                  # B
#         self.y_dim = y_dim                            # M
#         self.h_dim = h_dim                            # H
#         self.d_model = d_model                        # D
#         self.n_heads = n_heads
#         self.n_layers = n_layers
#         self.dim_feedforward = dim_feedforward
#         self.time_dim = time_dim
#         self.num_condition_tokens = num_condition_tokens  # K

#         factory_kwargs = {
#             "device": device,
#             "dtype": dtype,
#         }

#         # ------------------------------------------------------------
#         # 1. Convert scalar label coordinates into label tokens.
#         # ------------------------------------------------------------
#         #
#         # Input:
#         #     u_t_scalar: [B, M, 1]
#         #
#         # Output:
#         #     value_tokens: [B, M, D]

#         self.value_projection = nn.Linear(
#             in_features=1,
#             out_features=d_model,
#             **factory_kwargs,
#         )

#         # Learned label-coordinate identity embedding.
#         #
#         # Shape:
#         #     label_position_embedding: [1, M, D]
#         #
#         # This tells the Transformer which coordinate each token represents.
#         self.label_position_embedding = nn.Parameter(
#             0.02 * torch.randn(
#                 1,
#                 y_dim,
#                 d_model,
#                 **factory_kwargs,
#             )
#         )

#         # ------------------------------------------------------------
#         # 2. Time embedding.
#         # ------------------------------------------------------------

#         self.time_embedding = SinusoidalTimeEmbedding(
#             batch_size=batch_size,
#             time_dim=time_dim,
#             dtype=dtype,
#             device=device,
#         )

#         # Input:
#         #     time_emb: [B, time_dim]
#         #
#         # Output:
#         #     time_token: [B, D]
#         self.time_projection = nn.Sequential(
#             nn.Linear(time_dim, d_model, **factory_kwargs),
#             nn.SiLU(),
#             nn.Linear(d_model, d_model, **factory_kwargs),
#         )

#         # ------------------------------------------------------------
#         # 3. Convert global image embedding h into condition memory.
#         # ------------------------------------------------------------
#         #
#         # Input:
#         #     h: [B, H_img]
#         #
#         # Output:
#         #     memory_flat: [B, K * D]
#         #     memory:      [B, K, D]
#         #
#         # K is num_condition_tokens.
#         # These K tokens are learned condition slots derived from the single
#         # global image embedding.

#         self.condition_projection = nn.Sequential(
#             nn.LayerNorm(h_dim, **factory_kwargs),
#             nn.Linear(
#                 h_dim,
#                 num_condition_tokens * d_model,
#                 **factory_kwargs,
#             ),
#         )

#         # ------------------------------------------------------------
#         # 4. Transformer decoder.
#         # ------------------------------------------------------------
#         #
#         # tgt:
#         #     label_tokens: [B, M, D]
#         #
#         # memory:
#         #     image condition memory: [B, K, D]
#         #
#         # The decoder performs:
#         #     self-attention over label tokens
#         #     cross-attention from label tokens to image memory
#         #     feedforward updates

#         decoder_layer = nn.TransformerDecoderLayer(
#             d_model=d_model,
#             nhead=n_heads,
#             dim_feedforward=dim_feedforward,
#             dropout=dropout,
#             activation="gelu",
#             batch_first=True,
#             norm_first=True,
#             **factory_kwargs,
#         )

#         self.transformer = nn.TransformerDecoder(
#             decoder_layer=decoder_layer,
#             num_layers=n_layers,
#         )

#         # ------------------------------------------------------------
#         # 5. Output projection.
#         # ------------------------------------------------------------
#         #
#         # Input:
#         #     hidden: [B, M, D]
#         #
#         # Output:
#         #     pred_v_token: [B, M, 1]
#         #
#         # Final:
#         #     pred_v: [B, M]

#         self.output_norm = nn.LayerNorm(d_model, **factory_kwargs)

#         self.output_projection = nn.Linear(
#             in_features=d_model,
#             out_features=1,
#             **factory_kwargs,
#         )

#     def forward(
#         self,
#         u_t: torch.Tensor,
#         t: torch.Tensor,
#         h: torch.Tensor,
#     ) -> torch.Tensor:
#         """
#         Parameters
#         ----------
#         u_t:
#             Current interpolated label state.
#             Shape: [B, M]

#         t:
#             Flow time.
#             Shape: [B, 1]

#         h:
#             Global image embedding from encoder(X).
#             Shape: [B, H_img]

#         Returns
#         -------
#         pred_v:
#             Predicted velocity in label space.
#             Shape: [B, M]
#         """

#         # ------------------------------------------------------------
#         # 0. Fixed-shape validation.
#         # ------------------------------------------------------------

#         # u_t: [B, M]
#         assert u_t.ndim == 2
#         assert u_t.shape == (self.batch_size, self.y_dim)

#         # t: [B, 1]
#         assert t.ndim == 2
#         assert t.shape == (self.batch_size, 1)

#         # h: [B, H_img]
#         assert h.ndim == 2
#         assert h.shape == (self.batch_size, self.h_dim)

#         # Enforce shared dtype/device.
#         assert t.dtype == u_t.dtype
#         assert h.dtype == u_t.dtype

#         assert t.device == u_t.device
#         assert h.device == u_t.device

#         # Also check consistency with model parameters.
#         assert u_t.dtype == self.label_position_embedding.dtype
#         assert u_t.device == self.label_position_embedding.device

#         B = self.batch_size       # scalar
#         M = self.y_dim            # scalar
#         D = self.d_model          # scalar
#         K = self.num_condition_tokens

#         # ------------------------------------------------------------
#         # 1. Build label tokens from u_t.
#         # ------------------------------------------------------------

#         # u_t_scalar: [B, M, 1]
#         u_t_scalar = u_t.unsqueeze(-1)

#         assert u_t_scalar.shape == (B, M, 1)

#         # value_tokens: [B, M, D]
#         value_tokens = self.value_projection(u_t_scalar)

#         assert value_tokens.shape == (B, M, D)

#         # label_position_embedding: [1, M, D]
#         assert self.label_position_embedding.shape == (1, M, D)

#         # label_tokens: [B, M, D]
#         label_tokens = value_tokens + self.label_position_embedding

#         assert label_tokens.shape == (B, M, D)

#         # ------------------------------------------------------------
#         # 2. Add time information.
#         # ------------------------------------------------------------

#         # time_emb: [B, time_dim]
#         time_emb = self.time_embedding(t)

#         assert time_emb.shape == (B, self.time_dim)

#         # time_token: [B, D]
#         time_token = self.time_projection(time_emb)

#         assert time_token.shape == (B, D)

#         # time_token: [B, 1, D]
#         time_token = time_token.unsqueeze(1)

#         assert time_token.shape == (B, 1, D)

#         # label_tokens: [B, M, D]
#         label_tokens = label_tokens + time_token

#         assert label_tokens.shape == (B, M, D)

#         # ------------------------------------------------------------
#         # 3. Build image condition memory.
#         # ------------------------------------------------------------

#         # h: [B, H_img]
#         assert h.shape == (B, self.h_dim)

#         # memory_flat: [B, K * D]
#         memory_flat = self.condition_projection(h)

#         assert memory_flat.shape == (B, K * D)

#         # memory: [B, K, D]
#         memory = memory_flat.reshape(B, K, D)

#         assert memory.shape == (B, K, D)

#         # ------------------------------------------------------------
#         # 4. Transformer decoder with cross-attention conditioning.
#         # ------------------------------------------------------------
#         #
#         # tgt:
#         #     label_tokens: [B, M, D]
#         #
#         # memory:
#         #     memory: [B, K, D]
#         #
#         # Output:
#         #     hidden: [B, M, D]

#         hidden = self.transformer(
#             tgt=label_tokens,
#             memory=memory,
#         )

#         assert hidden.shape == (B, M, D)

#         # ------------------------------------------------------------
#         # 5. Project each label token to one velocity scalar.
#         # ------------------------------------------------------------

#         # hidden: [B, M, D]
#         hidden = self.output_norm(hidden)

#         assert hidden.shape == (B, M, D)

#         # pred_v_token: [B, M, 1]
#         pred_v_token = self.output_projection(hidden)

#         assert pred_v_token.shape == (B, M, 1)

#         # pred_v: [B, M]
#         pred_v = pred_v_token.squeeze(-1)

#         assert pred_v.shape == (B, M)

#         return pred_v





# class PosteriorODEFunc(nn.Module):
#     """
#     ODE function wrapper for torchdiffeq.

#     Converts torchdiffeq's scalar time t into the fixed-shape
#     time tensor expected by PosteriorVectorFieldTransformer.

#     Input from torchdiffeq:
#         scalar_t: []
#         u:        [B, M]

#     Internal model call:
#         posterior_vector_field(
#             u_t=u,        # [B, M]
#             t=t_batch,    # [B, 1]
#             h=h           # [B, H_img]
#         )

#     Output:
#         du_dt: [B, M]
#     """

#     def __init__(
#         self,
#         posterior_vector_field: nn.Module,
#         h: torch.Tensor,
#     ):
#         super().__init__()

#         self.posterior_vector_field = posterior_vector_field
#         self.h = h

#         # Fixed dimensions from the trained vector field.
#         self.batch_size = posterior_vector_field.batch_size  # B
#         self.y_dim = posterior_vector_field.y_dim            # M

#         # h: [B, H_img]
#         assert h.ndim == 2
#         assert h.shape == (
#             posterior_vector_field.batch_size,
#             posterior_vector_field.h_dim,
#         )

#     def forward(
#         self,
#         scalar_t: torch.Tensor,
#         u: torch.Tensor,
#     ) -> torch.Tensor:
#         """
#         Parameters
#         ----------
#         scalar_t:
#             Scalar ODE time from torchdiffeq.
#             Shape: []

#         u:
#             Current label-space state.
#             Shape: [B, M]

#         Returns
#         -------
#         du_dt:
#             Velocity.
#             Shape: [B, M]
#         """

#         # u: [B, M]
#         assert u.ndim == 2
#         assert u.shape == (self.batch_size, self.y_dim)

#         # scalar_t: []
#         assert scalar_t.ndim == 0

#         # t_batch: [B, 1]
#         t_batch = scalar_t.expand(self.batch_size, 1)

#         assert t_batch.shape == (self.batch_size, 1)
#         assert t_batch.dtype == u.dtype
#         assert t_batch.device == u.device

#         # du_dt: [B, M]
#         du_dt = self.posterior_vector_field(
#             u_t=u,
#             t=t_batch,
#             h=self.h,
#         )

#         assert du_dt.shape == (self.batch_size, self.y_dim)

#         return du_dt

    


# class CNFFMModel(nn.Module):
#     """
#     Posterior flow matching model.
#     """

#     def __init__(self, encoder, y_dim, batch_size, vectorfield_kwargs):
#         super().__init__()

#         self.y_dim = y_dim
#         self.batch_size = batch_size
#         self.encoder = encoder
#         self.vector_field = PosteriorVectorFieldTransformer(y_dim=y_dim, batch_size=batch_size, h_dim=encoder.embed_dim, **vectorfield_kwargs)
#         self.loss = nn.MSELoss()

#     def forward(self, X, y):
#         """
#         X: input image data [B, N, C]
#         y: target vector labels [B, M]
#         """

#         #
#         # check inputs
#         #

#         # X: [B, N, C]
#         assert X.ndim == 3
#         assert X.shape[0] == self.batch_size

#         # y: [B, M]
#         assert y.ndim == 2
#         assert y.shape == (self.batch_size, self.y_dim)

#         # u1: [B, M]
#         # Assumes y is already normalized and unconstrained.
#         u1 = y

#         #
#         # encode image
#         #

#         # h: [B, H_img] 
#         h = self.encoder(X)

#         #
#         # generate t0 variables
#         # 
        
#         u0 = torch.randn_like(u1)
        
#         # t: [B, 1]
#         t = torch.rand(self.batch_size, 1, device=u1.device, dtype=u1.dtype)

#         # Avoid exact t = 0 or t = 1.
#         eps = 1e-4
#         t = eps + (1.0 - 2.0 * eps) * t

#         #
#         # Build interpolated state
#         #

#         # ut: [B, M]
#         # Linear interpolation between base noise u0 and data label u1.
#         ut = (1.0 - t) * u0 + t * u1


#         # Compute target velocity
        
#         # target_v: [B, M]
#         # For the linear path, du_t / dt = u1 - u0.
#         target_v = u1 - u0


#         # pred_v: [B, M]
#         pred_v = self.vector_field(u_t=ut, t=t, h=h)
#         assert pred_v.shape == (self.batch_size, self.y_dim)

#         #
#         # compute loss
#         #

#         # loss: scalar
#         loss = self.loss(pred_v, target_v)

#         return loss

#     @torch.no_grad()
#     def predict(self, X, num_samples: int = 4, method: Literal["dopri5", "rk4", "euler", "midpoint"] = "dopri5", rtol: float = 1e-5, atol: float = 1e-5):
#         """
#         Sample y ~ q_phi(y | X) using torchdiffeq.odeint.

#         Assumed fixed shapes:
#             X_batch: [B, C, N, N]
#             h:       [B, H_img]
#             u0:      [B, M]
#             u1:      [B, M]
#             y:       [B, M]

#         Returns
#         -------
#         y_batch: [B, M]

#         """

#         #
#         # check inputs
#         #

#         # X: [B, N, C]
#         assert X.ndim == 3
#         assert X.shape[0] == self.batch_size

#         self.encoder.eval()
#         self.vector_field.eval()
        
#         # 
#         # 1. Encode X once.
#         # 

#         # h: [B, H_img]
#         h = self.encoder(X)

#         # 
#         # 2. Build ODE function du/dt = v_phi(u, t, h).
#         # 

#         ode_func = PosteriorODEFunc(
#             posterior_vector_field=self.vector_field,
#             h=h,
#         )

#         # t_span: [2]
#         # Integrate from t = 0 to t = 1.
#         t_span = torch.tensor(
#             [0.0, 1.0],
#             device=X.device,
#             dtype=X.dtype,
#         )

#         all_samples = []
#         for _ in range(num_samples):

#             # 
#             # 3. Sample initial base noise.
#             # 

#             # u0: [B, M]
#             u0 = torch.randn(
#                 self.batch_size,
#                 self.y_dim,
#                 device=X.device,
#                 dtype=X.dtype,
#             )

#             # 
#             # 4. Solve ODE.
#             # 
#             #
#             # trajectory: [2, B, M]
#             #
#             # trajectory[0] is approximately u(t=0)
#             # trajectory[1] is approximately u(t=1)

#             trajectory = odeint(
#                 func=ode_func,
#                 y0=u0,
#                 t=t_span,
#                 method=method,
#                 rtol=rtol,
#                 atol=atol,
#             )

#             assert trajectory.shape == (2, self.batch_size, self.y_dim)

#             # u1: [B, M]
#             u1 = trajectory[-1]

#             assert u1.shape == (self.batch_size, self.y_dim)

#             all_samples.append(u1)

#         all_samples = torch.stack(all_samples, dim=0)

#         y_sample = torch.mean(all_samples, dim=0)

#         # [B, M]
#         return y_sample


