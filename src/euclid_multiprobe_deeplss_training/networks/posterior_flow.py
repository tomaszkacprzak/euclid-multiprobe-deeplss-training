import math
import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    """
    Fixed-shape sinusoidal embedding for scalar flow time t.

    Expected input:
        t: [B, 1]

    Output:
        emb: [B, time_dim]
    """

    def __init__(
        self,
        batch_size: int,
        time_dim: int,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ):
        super().__init__()

        assert batch_size > 0
        assert time_dim > 0
        assert time_dim % 2 == 0

        self.batch_size = batch_size        # B
        self.time_dim = time_dim            # time_dim

        half_dim = time_dim // 2            # time_dim / 2

        # frequencies: [time_dim // 2]
        frequencies = torch.exp(
            -math.log(10000.0)
            * torch.arange(half_dim, dtype=dtype, device=device)
            / max(half_dim - 1, 1)
        )

        # Stored as a buffer so it moves with model.to(device/dtype), but is
        # not a trainable parameter.
        self.register_buffer("frequencies", frequencies, persistent=False)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        t:
            Flow time.
            Shape: [B, 1]

        Returns
        -------
        emb:
            Sinusoidal time embedding.
            Shape: [B, time_dim]
        """

        # t: [B, 1]
        assert t.ndim == 2
        assert t.shape == (self.batch_size, 1)

        # Enforce dtype/device consistency with this module's buffer.
        assert t.dtype == self.frequencies.dtype
        assert t.device == self.frequencies.device

        # frequencies: [time_dim // 2]
        assert self.frequencies.shape == (self.time_dim // 2,)

        # args: [B, time_dim // 2]
        args = 2.0 * math.pi * t * self.frequencies[None, :]

        # sin_part: [B, time_dim // 2]
        sin_part = torch.sin(args)

        # cos_part: [B, time_dim // 2]
        cos_part = torch.cos(args)

        # emb: [B, time_dim]
        emb = torch.cat([sin_part, cos_part], dim=-1)

        assert emb.shape == (self.batch_size, self.time_dim)

        return emb


class PosteriorVectorFieldTransformer(nn.Module):
    """
    Fixed-shape Transformer posterior vector field.

    This implements:

        pred_v = v_phi(u_t, t, h)

    for posterior label flow matching:

        q_phi(y | X)

    Expected inputs:
        u_t: [B, M]
        t:   [B, 1]
        h:   [B, H_img]

    Output:
        pred_v: [B, M]

    Shape meanings:
        B      = fixed batch size
        M      = label dimension
        H_img  = image encoder output dimension
        D      = Transformer hidden dimension
        K      = number of learned condition-memory tokens
    """

    def __init__(
        self,
        batch_size: int,
        y_dim: int,
        h_img_dim: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        dim_feedforward: int = 512,
        time_dim: int = 64,
        num_condition_tokens: int = 4,
        dropout: float = 0.0,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ):
        super().__init__()

        assert batch_size > 0
        assert y_dim > 0
        assert h_img_dim > 0
        assert d_model > 0
        assert n_heads > 0
        assert d_model % n_heads == 0
        assert n_layers > 0
        assert dim_feedforward > 0
        assert time_dim > 0
        assert time_dim % 2 == 0
        assert num_condition_tokens > 0

        self.batch_size = batch_size                  # B
        self.y_dim = y_dim                            # M
        self.h_img_dim = h_img_dim                    # H_img
        self.d_model = d_model                        # D
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dim_feedforward = dim_feedforward
        self.time_dim = time_dim
        self.num_condition_tokens = num_condition_tokens  # K

        factory_kwargs = {
            "device": device,
            "dtype": dtype,
        }

        # ------------------------------------------------------------
        # 1. Convert scalar label coordinates into label tokens.
        # ------------------------------------------------------------
        #
        # Input:
        #     u_t_scalar: [B, M, 1]
        #
        # Output:
        #     value_tokens: [B, M, D]

        self.value_projection = nn.Linear(
            in_features=1,
            out_features=d_model,
            **factory_kwargs,
        )

        # Learned label-coordinate identity embedding.
        #
        # Shape:
        #     label_position_embedding: [1, M, D]
        #
        # This tells the Transformer which coordinate each token represents.
        self.label_position_embedding = nn.Parameter(
            0.02 * torch.randn(
                1,
                y_dim,
                d_model,
                **factory_kwargs,
            )
        )

        # ------------------------------------------------------------
        # 2. Time embedding.
        # ------------------------------------------------------------

        self.time_embedding = SinusoidalTimeEmbedding(
            batch_size=batch_size,
            time_dim=time_dim,
            dtype=dtype,
            device=device,
        )

        # Input:
        #     time_emb: [B, time_dim]
        #
        # Output:
        #     time_token: [B, D]
        self.time_projection = nn.Sequential(
            nn.Linear(time_dim, d_model, **factory_kwargs),
            nn.SiLU(),
            nn.Linear(d_model, d_model, **factory_kwargs),
        )

        # ------------------------------------------------------------
        # 3. Convert global image embedding h into condition memory.
        # ------------------------------------------------------------
        #
        # Input:
        #     h: [B, H_img]
        #
        # Output:
        #     memory_flat: [B, K * D]
        #     memory:      [B, K, D]
        #
        # K is num_condition_tokens.
        # These K tokens are learned condition slots derived from the single
        # global image embedding.

        self.condition_projection = nn.Sequential(
            nn.LayerNorm(h_img_dim, **factory_kwargs),
            nn.Linear(
                h_img_dim,
                num_condition_tokens * d_model,
                **factory_kwargs,
            ),
        )

        # ------------------------------------------------------------
        # 4. Transformer decoder.
        # ------------------------------------------------------------
        #
        # tgt:
        #     label_tokens: [B, M, D]
        #
        # memory:
        #     image condition memory: [B, K, D]
        #
        # The decoder performs:
        #     self-attention over label tokens
        #     cross-attention from label tokens to image memory
        #     feedforward updates

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
            **factory_kwargs,
        )

        self.transformer = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=n_layers,
        )

        # ------------------------------------------------------------
        # 5. Output projection.
        # ------------------------------------------------------------
        #
        # Input:
        #     hidden: [B, M, D]
        #
        # Output:
        #     pred_v_token: [B, M, 1]
        #
        # Final:
        #     pred_v: [B, M]

        self.output_norm = nn.LayerNorm(d_model, **factory_kwargs)

        self.output_projection = nn.Linear(
            in_features=d_model,
            out_features=1,
            **factory_kwargs,
        )

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
            Current interpolated label state.
            Shape: [B, M]

        t:
            Flow time.
            Shape: [B, 1]

        h:
            Global image embedding from image_encoder(X).
            Shape: [B, H_img]

        Returns
        -------
        pred_v:
            Predicted velocity in label space.
            Shape: [B, M]
        """

        # ------------------------------------------------------------
        # 0. Fixed-shape validation.
        # ------------------------------------------------------------

        # u_t: [B, M]
        assert u_t.ndim == 2
        assert u_t.shape == (self.batch_size, self.y_dim)

        # t: [B, 1]
        assert t.ndim == 2
        assert t.shape == (self.batch_size, 1)

        # h: [B, H_img]
        assert h.ndim == 2
        assert h.shape == (self.batch_size, self.h_img_dim)

        # Enforce shared dtype/device.
        assert t.dtype == u_t.dtype
        assert h.dtype == u_t.dtype

        assert t.device == u_t.device
        assert h.device == u_t.device

        # Also check consistency with model parameters.
        assert u_t.dtype == self.label_position_embedding.dtype
        assert u_t.device == self.label_position_embedding.device

        B = self.batch_size       # scalar
        M = self.y_dim            # scalar
        D = self.d_model          # scalar
        K = self.num_condition_tokens

        # ------------------------------------------------------------
        # 1. Build label tokens from u_t.
        # ------------------------------------------------------------

        # u_t_scalar: [B, M, 1]
        u_t_scalar = u_t.unsqueeze(-1)

        assert u_t_scalar.shape == (B, M, 1)

        # value_tokens: [B, M, D]
        value_tokens = self.value_projection(u_t_scalar)

        assert value_tokens.shape == (B, M, D)

        # label_position_embedding: [1, M, D]
        assert self.label_position_embedding.shape == (1, M, D)

        # label_tokens: [B, M, D]
        label_tokens = value_tokens + self.label_position_embedding

        assert label_tokens.shape == (B, M, D)

        # ------------------------------------------------------------
        # 2. Add time information.
        # ------------------------------------------------------------

        # time_emb: [B, time_dim]
        time_emb = self.time_embedding(t)

        assert time_emb.shape == (B, self.time_dim)

        # time_token: [B, D]
        time_token = self.time_projection(time_emb)

        assert time_token.shape == (B, D)

        # time_token: [B, 1, D]
        time_token = time_token.unsqueeze(1)

        assert time_token.shape == (B, 1, D)

        # label_tokens: [B, M, D]
        label_tokens = label_tokens + time_token

        assert label_tokens.shape == (B, M, D)

        # ------------------------------------------------------------
        # 3. Build image condition memory.
        # ------------------------------------------------------------

        # h: [B, H_img]
        assert h.shape == (B, self.h_img_dim)

        # memory_flat: [B, K * D]
        memory_flat = self.condition_projection(h)

        assert memory_flat.shape == (B, K * D)

        # memory: [B, K, D]
        memory = memory_flat.reshape(B, K, D)

        assert memory.shape == (B, K, D)

        # ------------------------------------------------------------
        # 4. Transformer decoder with cross-attention conditioning.
        # ------------------------------------------------------------
        #
        # tgt:
        #     label_tokens: [B, M, D]
        #
        # memory:
        #     memory: [B, K, D]
        #
        # Output:
        #     hidden: [B, M, D]

        hidden = self.transformer(
            tgt=label_tokens,
            memory=memory,
        )

        assert hidden.shape == (B, M, D)

        # ------------------------------------------------------------
        # 5. Project each label token to one velocity scalar.
        # ------------------------------------------------------------

        # hidden: [B, M, D]
        hidden = self.output_norm(hidden)

        assert hidden.shape == (B, M, D)

        # pred_v_token: [B, M, 1]
        pred_v_token = self.output_projection(hidden)

        assert pred_v_token.shape == (B, M, 1)

        # pred_v: [B, M]
        pred_v = pred_v_token.squeeze(-1)

        assert pred_v.shape == (B, M)

        return pred_v


class PosteriorFlowMatchingModel(nn.Module):
    """
    Posterior flow matching model.
    """

    def __init__(self, y_dim, batch_size, encoder, vectorfield_kwargs):
        super().__init__()

        self.y_dim = y_dim
        self.batch_size = batch_size
        self.encoder = encoder
        self.vector_field = PosteriorVectorFieldTransformer(y_dim=y_dim, batch_size=batch_size, **vectorfield_kwargs)

    def forward(self, X, y):

        #
        # check inputs
        #

        # X: [B, C, N, N]
        assert X.ndim == 4
        assert X.shape[0] == self.batch_size

        # y: [B, M]
        assert y.ndim == 2
        assert y.shape == (self.batch_size, self.y_dim)

        # u1: [B, M]
        # Assumes y is already normalized and unconstrained.
        u1 = y

        #
        # encode image
        #

        # h: [B, H_img] 
        h = self.encoder(X)

        #
        # generate time variables
        # 
        
        u0 = torch.randn_like(u1)
        
        # t: [B, 1]
        t = torch.rand(batch, 1, device=u1.device, dtype=u1.dtype)

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
        assert pred_v.shape == (self.batch_size, self.y_dim)
