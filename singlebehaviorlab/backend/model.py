import logging
import os
import gc

logger = logging.getLogger(__name__)

# JAX memory config MUST be set before importing jax.
# Without this, JAX grabs 75-90% of GPU memory upfront, leaving PyTorch
# starved and eventually causing CUDA_ERROR_ILLEGAL_ADDRESS under sustained
# inference workloads where both frameworks share the same GPU.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.45")

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable
import numpy as np
import jax
import jax.numpy as jnp
from videoprism import models as vp


def interpolate_pos_embed_2d(pos_embed: np.ndarray, orig_grid: int, new_grid: int) -> np.ndarray:
    """
    Bicubic interpolation of 2D spatial position embeddings.
    
    Args:
        pos_embed: [N, D] where N = orig_grid * orig_grid
        orig_grid: original spatial grid size (e.g., 16 for 288px)
        new_grid: target spatial grid size (e.g., 19 for 342px)
    
    Returns:
        Interpolated pos_embed: [new_grid*new_grid, D]
    """
    if orig_grid == new_grid:
        return pos_embed
    
    D = pos_embed.shape[-1]
    pos_2d = pos_embed.reshape(orig_grid, orig_grid, D)
    pos_torch = torch.from_numpy(pos_2d).permute(2, 0, 1).unsqueeze(0).float()
    
    pos_interp = F.interpolate(
        pos_torch,
        size=(new_grid, new_grid),
        mode='bicubic',
        align_corners=False
    )
    
    pos_interp = pos_interp.squeeze(0).permute(1, 2, 0).numpy()
    return pos_interp.reshape(new_grid * new_grid, D)


class VideoPrismBackbone(nn.Module):
    """VideoPrism backbone wrapper for PyTorch compatibility."""
    
    DEFAULT_RESOLUTION = 288
    PATCH_SIZE = 18
    
    def __init__(
        self,
        model_name: str = 'videoprism_public_v1_base',
        resolution: int = 288,
        log_fn: Optional[Callable[[str], None]] = None
    ):
        super().__init__()
        self.model_name = model_name
        self.resolution = resolution
        self.flax_model = None
        self.params = None
        self.original_params = None
        self.jax_device = None
        self._forward_fn = None
        self.log_fn = log_fn or print
        self._current_grid_size = None
        self.enable_dlpack = os.environ.get("BEHAVIOR_APP_USE_DLPACK", "0").strip().lower() in ("1", "true", "yes", "on")
        self._load_model()
    
    def _is_jax_gpu(self) -> bool:
        d = self.jax_device
        return (
            d.device_kind == 'gpu' or
            'gpu' in d.device_kind.lower() or
            'cuda' in str(d).lower() or
            d.platform in ['gpu', 'cuda']
        )

    def _load_model(self):
        """Load VideoPrism model (JAX/Flax) and configure for GPU if available."""
        try:
            import copy
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
            os.environ['XLA_FLAGS'] = '--xla_gpu_unsafe_fallback_to_driver_on_ptxas_not_found'
            
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                self.log_fn("Cleared PyTorch GPU cache before loading VideoPrism")
            
            devices = jax.devices()
            self.log_fn(f"JAX devices found: {len(devices)}")
            for i, d in enumerate(devices):
                self.log_fn(f"  Device {i}: {d} (kind: {d.device_kind}, platform: {d.platform})")
            
            gpu_devices = [
                d for d in devices 
                if (d.device_kind == 'gpu' or 
                    'gpu' in d.device_kind.lower() or 
                    'cuda' in str(d).lower() or
                    d.platform in ['gpu', 'cuda'])
            ]
            
            if gpu_devices:
                self.jax_device = gpu_devices[0]
                self.log_fn(f"JAX GPU device selected: {self.jax_device}")
            else:
                self.jax_device = jax.devices('cpu')[0]
                self.log_fn(f"JAX using CPU device: {self.jax_device} (GPU not available)")
            
            self.log_fn("Loading VideoPrism model...")
            self.flax_model = vp.get_model(self.model_name)
            self.params = vp.load_pretrained_weights(self.model_name)
            self.original_params = copy.deepcopy(self.params)
            
            orig_grid = self.DEFAULT_RESOLUTION // self.PATCH_SIZE
            new_grid = self.resolution // self.PATCH_SIZE
            self._current_grid_size = new_grid
            
            if new_grid != orig_grid:
                self.log_fn(f"Resolution {self.resolution}x{self.resolution} -> {new_grid}x{new_grid} spatial grid")
                self.params = self._interpolate_spatial_pos_embed(self.params, orig_grid, new_grid)
            else:
                self.log_fn(f"Using default resolution {self.resolution}x{self.resolution} ({new_grid}x{new_grid} grid)")
            
            if self._is_jax_gpu():
                self.log_fn("Moving VideoPrism parameters to GPU...")
                self.params = jax.device_put(self.params, self.jax_device)
            
            @jax.jit
            def forward_fn(params, videos_bthwc: jnp.ndarray) -> jnp.ndarray:
                embeddings, _ = self.flax_model.apply(
                    params, videos_bthwc, train=False, return_intermediate=False
                )
                return embeddings
            
            self._forward_fn = forward_fn
            self.log_fn(f"Loaded VideoPrism model: {self.model_name} (device: {self.jax_device.device_kind})")
            
            self.log_fn(f"Warming up JIT compilation at {self.resolution}x{self.resolution}...")
            dummy_batch = jnp.zeros((1, 16, self.resolution, self.resolution, 3), dtype=jnp.float32)
            dummy_batch = jax.device_put(dummy_batch, self.jax_device)
            _ = self._forward_fn(self.params, dummy_batch)
            self.log_fn("JIT compilation complete")
            
        except Exception as e:
            error_msg = f"Error loading VideoPrism model: {e}\n"
            error_msg += "This may be due to CuDNN version mismatch or missing dependencies."
            if "DNN library initialization failed" in str(e) or "FAILED_PRECONDITION" in str(e):
                error_msg += (
                    "\n\nTry:\n"
                    "conda activate singlebehaviorlab\n"
                    "pip install --upgrade torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124\n"
                    "pip install --upgrade \"jax[cuda12]==0.6.2\" flax==0.10.7\n"
                    "pip install --upgrade \"nvidia-cudnn-cu12==9.20.0.48\""
                )
            self.log_fn(error_msg)
            import traceback
            self.log_fn(traceback.format_exc())
            raise RuntimeError(error_msg) from e
    
    def _interpolate_spatial_pos_embed(self, params: dict, orig_grid: int, new_grid: int) -> dict:
        import copy
        params = copy.deepcopy(params)
        
        def find_and_interpolate(d, path=""):
            if isinstance(d, dict):
                for k, v in d.items():
                    new_path = f"{path}.{k}" if path else k
                    d[k] = find_and_interpolate(v, new_path)
            elif isinstance(d, (np.ndarray, jnp.ndarray)):
                arr = np.asarray(d)
                expected_spatial = orig_grid * orig_grid
                is_pos_embed = any(name in path.lower() for name in ['pos_embed', 'posembed', 'position'])
                
                if is_pos_embed and arr.ndim >= 2:
                    if arr.shape[-2] == expected_spatial:
                        D = arr.shape[-1]
                        self.log_fn(f"  Interpolating {path}: [{expected_spatial}, {D}] -> [{new_grid*new_grid}, {D}]")
                        arr_interp = interpolate_pos_embed_2d(arr, orig_grid, new_grid)
                        return jnp.asarray(arr_interp)
                    elif arr.ndim == 3 and arr.shape[1] == expected_spatial:
                        D = arr.shape[-1]
                        self.log_fn(f"  Interpolating {path}: [1, {expected_spatial}, {D}] -> [1, {new_grid*new_grid}, {D}]")
                        arr_interp = interpolate_pos_embed_2d(arr[0], orig_grid, new_grid)
                        return jnp.asarray(arr_interp[np.newaxis, :, :])
                    elif arr.ndim == 3 and arr.shape[1] == expected_spatial + 1:
                        D = arr.shape[-1]
                        self.log_fn(f"  Interpolating {path}: [1, 1+{expected_spatial}, {D}] -> [1, 1+{new_grid*new_grid}, {D}]")
                        cls_token = arr[:, :1, :]
                        spatial = arr[0, 1:, :]
                        spatial_interp = interpolate_pos_embed_2d(spatial, orig_grid, new_grid)
                        return jnp.concatenate([cls_token, spatial_interp[np.newaxis, :, :]], axis=1)
                return d
            return d
        
        params = find_and_interpolate(params)
        return params
    
    def set_resolution(self, resolution: int):
        if resolution == self.resolution:
            return
        self.resolution = resolution
        orig_grid = self.DEFAULT_RESOLUTION // self.PATCH_SIZE
        new_grid = resolution // self.PATCH_SIZE
        self._current_grid_size = new_grid
        
        if self.original_params is not None:
            self.log_fn(f"Re-interpolating spatial pos embeddings for resolution {resolution}...")
            import copy
            self.params = copy.deepcopy(self.original_params)
            if new_grid != orig_grid:
                self.params = self._interpolate_spatial_pos_embed(self.params, orig_grid, new_grid)
            if self._is_jax_gpu():
                self.params = jax.device_put(self.params, self.jax_device)
            self.log_fn(f"Resolution updated to {resolution}x{resolution}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, T, C, H, W] in [0, 1] float32
        Returns:
            Token embeddings [B, N, D] where N = T * S spatial tokens
        """
        if self.flax_model is None or self.params is None or self._forward_fn is None:
            raise RuntimeError("Model not loaded")
        
        device = x.device
        x_bthwc = x.permute(0, 1, 3, 4, 2).contiguous()
        
        use_dlpack = (
            self.enable_dlpack and
            self.resolution <= 288 and
            x_bthwc.is_cuda and
            self.jax_device is not None and
            self.jax_device.platform in ["gpu", "cuda"]
        )

        def _to_jax_tensor(prefer_dlpack: bool):
            if prefer_dlpack:
                try:
                    x_jax_local = jax.dlpack.from_dlpack(x_bthwc)
                except Exception:
                    import torch.utils.dlpack as torch_dlpack
                    x_dlpack = torch_dlpack.to_dlpack(x_bthwc)
                    x_jax_local = jax.dlpack.from_dlpack(x_dlpack)
                if x_jax_local.device != self.jax_device:
                    x_jax_local = jax.device_put(x_jax_local, self.jax_device)
                return x_jax_local, True
            x_np = x_bthwc.detach().cpu().contiguous().numpy()
            x_jax_local = jnp.asarray(x_np)
            x_jax_local = jax.device_put(x_jax_local, self.jax_device)
            return x_jax_local, False

        try:
            x_jax, used_dlpack = _to_jax_tensor(use_dlpack)
            embeddings_jax = self._forward_fn(self.params, x_jax)
            embeddings_jax.block_until_ready()
            del x_jax
        except Exception as e:
            if use_dlpack:
                self.log_fn(f"Warning: DLPack path failed ({e}). Retrying with safe host-copy transfer.")
                x_jax, used_dlpack = _to_jax_tensor(False)
                embeddings_jax = self._forward_fn(self.params, x_jax)
                embeddings_jax.block_until_ready()
                del x_jax
            else:
                raise

        embeddings_np = np.asarray(embeddings_jax)
        del embeddings_jax
        embeddings_torch = torch.from_numpy(embeddings_np.copy()).to(device)
        del embeddings_np
        return embeddings_torch
    
    def get_embed_dim(self) -> int:
        dummy_input = torch.zeros(1, 16, 3, self.resolution, self.resolution)
        with torch.no_grad():
            tokens = self.forward(dummy_input)
        return tokens.shape[-1]
    
    def get_num_tokens(self) -> int:
        grid_size = self.resolution // self.PATCH_SIZE
        return grid_size * grid_size


# Stage B: frame decoder and localization heads.

class SpatialAttentionPool(nn.Module):
    """Trainable per-frame spatial attention pooling.
    
    For each frame, a learned query attends over S spatial tokens
    and an MLP bottleneck projects the result to proj_dim.
    """

    def __init__(self, embed_dim: int, proj_dim: int = 256,
                 num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.proj_dim = proj_dim
        self.ln = nn.LayerNorm(embed_dim)
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True,
        )
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim, proj_dim),
            nn.Dropout(dropout),
        )

    def forward(self, tokens: torch.Tensor, num_frames: int,
                return_attn_weights: bool = False):
        """
        Args:
            tokens: [B, T*S, D]
            num_frames: T
            return_attn_weights: if True, also return spatial attention maps
        Returns:
            [B, T, proj_dim]  or  ([B, T, proj_dim], [B, T, num_heads, S])
        """
        B, N, D = tokens.shape
        S = N // num_frames
        T = num_frames

        flat = tokens.view(B, T, S, D).reshape(B * T, S, D)
        flat = self.ln(flat)

        q = self.query.expand(B * T, -1, -1)
        pooled, attn_w = self.attn(q, flat, flat, need_weights=return_attn_weights,
                                   average_attn_weights=False)
        pooled = pooled.squeeze(1)  # [B*T, D]

        out = self.mlp(pooled)  # [B*T, proj_dim]

        if return_attn_weights and attn_w is not None:
            # attn_w: [B*T, num_heads, 1, S] -> [B, T, num_heads, S]
            attn_w = attn_w.squeeze(2).view(B, T, -1, S)
            return out.view(B, T, self.proj_dim), attn_w

        return out.view(B, T, self.proj_dim)


class _DilatedResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, k: int, p: float):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=k,
                      padding=dilation, dilation=dilation, bias=False),
            nn.GELU(),
            nn.Dropout(p),
            nn.Conv1d(channels, channels, kernel_size=1, bias=False),
            nn.Dropout(p),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class _MSRefineStage(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int,
                 out_channels: int, layers: int, k: int, p: float):
        super().__init__()
        self.in_proj = nn.Conv1d(in_channels, hidden_channels,
                                 kernel_size=1, bias=False)
        self.blocks = nn.ModuleList([
            _DilatedResidualBlock(hidden_channels, dilation=2 ** i, k=k, p=p)
            for i in range(max(1, int(layers)))
        ])
        self.out_proj = nn.Conv1d(hidden_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.in_proj(x)
        for blk in self.blocks:
            y = blk(y)
        return self.out_proj(y)


class DilatedTemporalHead(nn.Module):
    """Frame decoder head (Stage B).
    
    Trainable spatial attention pooling -> local TCN -> state + boundary heads.
    Produces per-frame state logits, boundary logits, and frame embeddings.
    """

    def __init__(
        self,
        embed_dim: int,
        num_classes: int,
        num_layers: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.2,
        num_stages: int = 3,
        hidden_dim: Optional[int] = None,
        temporal_pool: int = 1,
        proj_dim: int = 256,
        spatial_pool_heads: int = 4,
        multi_scale: bool = False,
        use_temporal_decoder: bool = True,
    ):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.num_classes = int(num_classes)
        self.num_layers = max(1, int(num_layers))
        self.num_stages = max(1, int(num_stages))
        self.dropout = float(dropout)
        self.temporal_pool = max(1, int(temporal_pool))
        self.use_ovr = False
        self.proj_dim = int(proj_dim)
        self.multi_scale = bool(multi_scale)
        self.use_temporal_decoder = bool(use_temporal_decoder)
        auto_hidden = max(128, self.proj_dim)
        self.hidden_dim = int(hidden_dim) if hidden_dim is not None else auto_hidden

        self.spatial_pool = SpatialAttentionPool(
            embed_dim=self.embed_dim,
            proj_dim=self.proj_dim,
            num_heads=int(spatial_pool_heads),
            dropout=self.dropout,
        )

        # Raw per-frame projection: simple mean-pool of spatial tokens → proj_dim.
        # Not filtered by learned attention, so it retains broader spatial context
        # that SpatialAttentionPool may suppress in favour of discriminative patches.
        self.raw_proj = nn.Linear(self.embed_dim, self.proj_dim)

        # When multi_scale=True, long-scale and short-scale features are concatenated,
        # so the TCN input is 2*proj_dim instead of proj_dim.
        tcn_in = self.proj_dim * 2 if self.multi_scale else self.proj_dim
        if self.use_temporal_decoder:
            self.stage1 = _MSRefineStage(
                in_channels=tcn_in,
                hidden_channels=self.hidden_dim,
                out_channels=self.num_classes,
                layers=self.num_layers,
                k=kernel_size,
                p=self.dropout,
            )
            self.refine_stages = nn.ModuleList()
            for _ in range(self.num_stages - 1):
                self.refine_stages.append(
                    _MSRefineStage(
                        in_channels=self.num_classes,
                        hidden_channels=self.hidden_dim,
                        out_channels=self.num_classes,
                        layers=self.num_layers,
                        k=kernel_size,
                        p=self.dropout,
                    )
                )

            self.boundary_tcn = _MSRefineStage(
                in_channels=tcn_in,
                hidden_channels=self.hidden_dim,
                out_channels=1,
                layers=max(1, self.num_layers // 2),
                k=kernel_size,
                p=self.dropout,
            )
            self.frame_classifier = None
        else:
            self.stage1 = None
            self.refine_stages = nn.ModuleList()
            self.boundary_tcn = None
            self.frame_classifier = nn.Sequential(
                nn.LayerNorm(tcn_in),
                nn.Linear(tcn_in, self.num_classes),
            )

    def forward(self, tokens: torch.Tensor, num_frames: int,
                tokens_short: Optional[torch.Tensor] = None,
                num_frames_short: Optional[int] = None,
                return_attn_weights: bool = False):
        """
        Args:
            tokens:            [B, T*S, D] long-scale backbone tokens
            num_frames:        T (long scale)
            tokens_short:      [B, T_s*S, D] short-scale tokens (half fps, T_s = T//2).
                               Required when multi_scale=True.
            num_frames_short:  T_s
            return_attn_weights: if True, include spatial attention maps in output
        Returns:
            frame_logits:        [B, T, C]
            clip_logits:         [B, C]
            temporal_weights:    [B, T]
            frame_logits_pooled: [B, T_pooled, C]
            boundary_logits:     [B, T, 1]
            frame_embeddings:    [B, T, proj_dim]  (long-scale attention pool)
            frame_embeddings_combined: [B, T, 2*proj_dim]  (attn || raw_mean, long scale)
            attn_weights:        [B, T, num_heads, S] or None
        """
        B, N, D = tokens.shape
        if num_frames <= 0 or (N % num_frames) != 0:
            raise ValueError(
                f"DilatedTemporalHead expects T*S tokens. "
                f"Got N={N}, num_frames={num_frames}."
            )
        T = num_frames

        # 1. Spatial attention pooling (long scale): [B, T*S, D] -> [B, T, proj_dim]
        attn_weights = None
        pool_out = self.spatial_pool(tokens, T, return_attn_weights=return_attn_weights)
        if return_attn_weights and isinstance(pool_out, tuple):
            x_long, attn_weights = pool_out
        else:
            x_long = pool_out
        # frame_embeddings is the long-scale per-frame feature
        frame_embeddings = x_long  # [B, T, proj_dim]

        # Multi-scale: pool short-scale tokens and upsample to align with T
        if self.multi_scale:
            if tokens_short is None or num_frames_short is None:
                raise ValueError(
                    "DilatedTemporalHead has multi_scale=True but tokens_short / "
                    "num_frames_short were not provided. Ensure the dataset has "
                    "_emb_multi_scale=True and short-scale embeddings are cached."
                )
            T_short = num_frames_short
            x_short = self.spatial_pool(tokens_short, T_short)  # [B, T_short, proj_dim]
            scale_factor = T // T_short
            # Nearest-neighbour upsample: each short frame covers scale_factor long frames
            x_short_up = x_short.repeat_interleave(scale_factor, dim=1)[:, :T, :]
            x = torch.cat([x_long, x_short_up], dim=-1)  # [B, T, 2*proj_dim]
        else:
            x = x_long  # [B, T, proj_dim]

        if self.use_temporal_decoder:
            # 2. Boundary prediction at full T (before temporal pooling reduces resolution)
            emb_conv = x.transpose(1, 2)  # [B, tcn_in, T]
            boundary_logits = self.boundary_tcn(emb_conv).transpose(1, 2)  # [B, T, 1]

            T_pooled = T
            if self.temporal_pool > 1:
                p = self.temporal_pool
                pad = (p - T % p) % p
                if pad > 0:
                    x = torch.cat([x, x[:, -1:, :].expand(-1, pad, -1)], dim=1)
                T_padded = T + pad
                feat_dim = x.shape[-1]
                x = x.view(B, T_padded // p, p, feat_dim).mean(dim=2)
                T_pooled = T_padded // p

            x_conv = x.transpose(1, 2)  # [B, tcn_in, T_pooled]
            stage_logits = self.stage1(x_conv)
            for refine in self.refine_stages:
                if self.use_ovr:
                    refine_in = torch.sigmoid(stage_logits)
                else:
                    refine_in = torch.softmax(stage_logits, dim=1)
                stage_logits = stage_logits + refine(refine_in)

            stage_logits_pooled = stage_logits
            clip_logits = stage_logits_pooled.mean(dim=2)

            if self.temporal_pool > 1:
                stage_logits = stage_logits.repeat_interleave(
                    self.temporal_pool, dim=2
                )[:, :, :T]

            frame_logits = stage_logits.transpose(1, 2)
            frame_logits_pooled = stage_logits_pooled.transpose(1, 2)
        else:
            boundary_logits = None
            frame_logits = self.frame_classifier(x)
            frame_logits_pooled = frame_logits
            clip_logits = frame_logits.mean(dim=1)
        temporal_weights = torch.ones(B, T, device=tokens.device) / T

        # Raw mean-pool per frame from long-scale tokens: bypasses spatial attention
        # so downstream analysis can access a less task-biased view as well.
        S = N // T
        raw_mean = tokens.view(B, T, S, self.embed_dim).mean(dim=2)
        raw_emb = self.raw_proj(raw_mean)  # [B, T, proj_dim]
        # Combined embedding: [attention_emb || raw_emb], shape [B, T, 2*proj_dim]
        frame_embeddings_combined = torch.cat([frame_embeddings, raw_emb], dim=-1)

        return (frame_logits, clip_logits, temporal_weights,
                frame_logits_pooled, boundary_logits, frame_embeddings,
                frame_embeddings_combined, attn_weights)


# Spatial localization head.

class SpatialLocalizationHead(nn.Module):
    """Frame-level bbox regressor over spatial tokens.

    Design:
    1) Dense objectness map over spatial patches per frame.
    2) Upsampled objectness for sub-patch precision.
    3) Temperature-scaled soft-argmax for coarse center.
    4) Local token gathering for context-aware refinement.
    5) Temporal residual refiner over center trajectories.
    6) Fixed box size from dataset stats for stable crops.
    """

    UPSAMPLE_FACTOR = 4

    def __init__(self, embed_dim: int, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads=4, dropout=dropout, batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(embed_dim)
        self.objectness = nn.Linear(embed_dim, 1)
        self.temperature = nn.Parameter(torch.tensor(8.0))

        self.refine = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2),
        )

        temporal_hidden = max(16, hidden_dim // 8)
        self.temporal_refine = nn.Sequential(
            nn.Conv1d(2, temporal_hidden, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(temporal_hidden, 2, kernel_size=5, padding=2),
        )
        final_temporal = self.temporal_refine[-1]
        if isinstance(final_temporal, nn.Conv1d):
            nn.init.zeros_(final_temporal.weight)
            if final_temporal.bias is not None:
                nn.init.zeros_(final_temporal.bias)

        proj_dim = hidden_dim // 2
        self.contrastive_proj = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, proj_dim),
        )

        self.register_buffer(
            "fixed_box_wh", torch.tensor([0.2, 0.2], dtype=torch.float32)
        )

    def set_fixed_box_size(self, width: float, height: float):
        w = float(max(1e-4, min(1.0, width)))
        h = float(max(1e-4, min(1.0, height)))
        self.fixed_box_wh.data = torch.tensor(
            [w, h], dtype=self.fixed_box_wh.dtype, device=self.fixed_box_wh.device,
        )

    def get_contrastive_tokens(
        self, tokens: torch.Tensor, num_frames: Optional[int] = None
    ) -> torch.Tensor:
        B, N, D = tokens.shape
        normed = self.norm(tokens)
        S = N
        if num_frames is not None and int(num_frames) > 1 and N % int(num_frames) == 0:
            S = N // int(num_frames)
        first_frame = normed[:, :S, :]
        return self.contrastive_proj(first_frame)

    def get_objectness_logits(
        self, tokens: torch.Tensor, num_frames: Optional[int] = None,
        all_frames: bool = False,
    ) -> torch.Tensor:
        B, N, D = tokens.shape
        normed = self.norm(tokens)
        S = N
        T = 1
        if num_frames is not None and int(num_frames) > 1 and N % int(num_frames) == 0:
            T = int(num_frames)
            S = N // T
        if all_frames and T > 1:
            frames = normed.view(B, T, S, D).reshape(B * T, S, D)
            return self.objectness(frames).squeeze(-1)
        else:
            first_frame = normed[:, :S, :]
            return self.objectness(first_frame).squeeze(-1)

    def forward(
        self,
        tokens: torch.Tensor,
        num_frames: Optional[int] = None,
        fixed_box_wh: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N, D = tokens.shape
        normed = self.norm(tokens)

        T = 1
        S = N
        if num_frames is not None and int(num_frames) > 1 and N % int(num_frames) == 0:
            T = int(num_frames)
            S = N // T
            frames_tokens = normed.view(B, T, S, D)
        else:
            frames_tokens = normed.view(B, 1, N, D)

        flat_tokens = frames_tokens.reshape(B * T, S, D)
        q = self.query.expand(B * T, -1, -1)
        pooled, _ = self.attn(q, flat_tokens, flat_tokens)
        pooled = self.attn_norm(pooled.squeeze(1))

        obj_logits = self.objectness(flat_tokens).squeeze(-1)

        g = int(round(S ** 0.5))
        is_square = g * g == S
        min_size = 1.0 / float(g) if is_square else 1.0 / float(max(1, S))

        if is_square:
            g_up = g * self.UPSAMPLE_FACTOR
            obj_2d = obj_logits.view(B * T, 1, g, g)
            obj_up = F.interpolate(obj_2d, size=(g_up, g_up), mode="bilinear", align_corners=False)
            obj_up = obj_up.view(B * T, g_up * g_up)

            temp = self.temperature.clamp(min=1.0)
            obj_probs = torch.softmax(obj_up * temp, dim=1)

            ys = (torch.arange(g_up, device=normed.device, dtype=normed.dtype) + 0.5) / float(g_up)
            xs = (torch.arange(g_up, device=normed.device, dtype=normed.dtype) + 0.5) / float(g_up)
            yy, xx = torch.meshgrid(ys, xs, indexing="ij")
            x_coords = xx.reshape(-1).unsqueeze(0)
            y_coords = yy.reshape(-1).unsqueeze(0)
        else:
            temp = self.temperature.clamp(min=1.0)
            obj_probs = torch.softmax(obj_logits * temp, dim=1)
            x_coords = ((torch.arange(S, device=normed.device, dtype=normed.dtype) + 0.5) / float(S)).unsqueeze(0)
            y_coords = torch.full_like(x_coords, 0.5)

        cx0 = (obj_probs * x_coords).sum(dim=1)
        cy0 = (obj_probs * y_coords).sum(dim=1)

        if is_square:
            cx0_grid = (cx0 * g).clamp(0, g - 1).long()
            cy0_grid = (cy0 * g).clamp(0, g - 1).long()
            tokens_2d = flat_tokens.view(B * T, g, g, D)
            local_feats = []
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    gy = (cy0_grid + dy).clamp(0, g - 1)
                    gx = (cx0_grid + dx).clamp(0, g - 1)
                    idx_bt = torch.arange(B * T, device=normed.device)
                    local_feats.append(tokens_2d[idx_bt, gy, gx, :])
            local_pool = torch.stack(local_feats, dim=1).mean(dim=1)
        else:
            local_pool = pooled

        refine_input = torch.cat([pooled, local_pool], dim=1)
        delta = self.refine(refine_input)
        dx = 0.30 * torch.tanh(delta[:, 0])
        dy = 0.30 * torch.tanh(delta[:, 1])

        cx = (cx0 + dx).clamp(0.0, 1.0)
        cy = (cy0 + dy).clamp(0.0, 1.0)

        if T > 1:
            center_seq = torch.stack([cx, cy], dim=1).view(B, T, 2).transpose(1, 2)
            temporal_delta = self.temporal_refine(center_seq).transpose(1, 2).reshape(B * T, 2)
            cx = (cx + 0.12 * torch.tanh(temporal_delta[:, 0])).clamp(0.0, 1.0)
            cy = (cy + 0.12 * torch.tanh(temporal_delta[:, 1])).clamp(0.0, 1.0)

        if fixed_box_wh is not None:
            wh = fixed_box_wh.to(device=normed.device, dtype=normed.dtype)
            if wh.dim() == 1:
                wh = wh.view(1, 2).expand(B, -1)
            if wh.size(0) != B:
                wh = wh[:B]
        else:
            wh = self.fixed_box_wh.to(device=normed.device, dtype=normed.dtype).view(1, 2).expand(B, -1)
        wh = wh.clamp(min=min_size, max=1.0)
        wh_bt = wh.unsqueeze(1).expand(B, T, 2).reshape(B * T, 2) if T > 1 else wh
        w = wh_bt[:, 0]
        h = wh_bt[:, 1]

        x1 = (cx - 0.5 * w).clamp(0.0, 1.0)
        y1 = (cy - 0.5 * h).clamp(0.0, 1.0)
        x2 = (cx + 0.5 * w).clamp(0.0, 1.0)
        y2 = (cy + 0.5 * h).clamp(0.0, 1.0)
        x2 = torch.maximum(x2, x1 + min_size).clamp(0.0, 1.0)
        y2 = torch.maximum(y2, y1 + min_size).clamp(0.0, 1.0)

        boxes = torch.stack([x1, y1, x2, y2], dim=1).view(B, T, 4)
        if T == 1:
            return boxes[:, 0, :]
        return boxes


# BehaviorClassifier.

class BehaviorClassifier(nn.Module):
    """Complete model: VideoPrism backbone + frame decoder head (Stage B)."""

    def __init__(
        self,
        backbone: VideoPrismBackbone,
        num_classes: int,
        class_names: list = None,
        dropout: float = 0.1,
        freeze_backbone: bool = True,
        head_kwargs: Optional[dict] = None,
        use_localization: bool = False,
        localization_hidden_dim: int = 256,
        localization_dropout: float = 0.0,
        frame_head_temporal_layers: int = 1,
        temporal_pool_frames: int = 1,
        proj_dim: int = 256,
        num_stages: int = 3,
        multi_scale: bool = False,
        use_temporal_decoder: bool = True,
        use_frame_head: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.backbone = backbone
        self.multi_scale = bool(multi_scale)
        self.use_temporal_decoder = bool(use_temporal_decoder)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        embed_dim = backbone.get_embed_dim()
        self.temporal_pool_frames = max(1, int(temporal_pool_frames))
        self.use_frame_head = True  # always on

        head_kwargs = head_kwargs or {}
        spatial_pool_heads = head_kwargs.get("num_heads", 4)

        self.frame_head = DilatedTemporalHead(
            embed_dim=embed_dim,
            num_classes=num_classes,
            dropout=dropout,
            num_layers=max(1, int(frame_head_temporal_layers)),
            num_stages=max(1, int(num_stages)),
            temporal_pool=self.temporal_pool_frames,
            proj_dim=int(proj_dim),
            spatial_pool_heads=int(spatial_pool_heads),
            multi_scale=self.multi_scale,
            use_temporal_decoder=self.use_temporal_decoder,
        )

        self.use_localization = bool(use_localization)
        self.localization_head = None
        if self.use_localization:
            self.localization_head = SpatialLocalizationHead(
                embed_dim=embed_dim,
                hidden_dim=int(localization_hidden_dim),
                dropout=float(localization_dropout),
            )

    def forward(
        self,
        video: Optional[torch.Tensor],
        return_localization: bool = False,
        return_frame_logits: bool = False,
        cache_backbone_tokens: bool = False,
        localization_box_wh: Optional[torch.Tensor] = None,
        return_attn_weights: bool = False,
        backbone_tokens: Optional[torch.Tensor] = None,
        num_frames: Optional[int] = None,
        backbone_tokens_short: Optional[torch.Tensor] = None,
        num_frames_short: Optional[int] = None,
        return_features: bool = False,
    ):
        """
        Returns:
            clip_logits [B, C] by default.
            Stores full frame outputs in self._frame_output when
            return_frame_logits=True.

        backbone_tokens: pre-computed tokens [B, T*S, D]. If provided,
            the backbone call is skipped. num_frames must also be provided.
        backbone_tokens_short: pre-computed short-scale tokens [B, T_s*S, D]
            (T_s = T//2, half-fps clip). Used when multi_scale=True.
        """
        if backbone_tokens is not None:
            # Embedding-space stitch path: bypass backbone entirely
            tokens_full = backbone_tokens
            if num_frames is None:
                raise ValueError("num_frames must be provided when backbone_tokens is given")
            T = num_frames
            tokens_short = backbone_tokens_short
            T_short = num_frames_short
        else:
            tokens_full = self.backbone(video)
            T = int(video.shape[1])
            # Multi-scale at inference: subsample video by 2 and run backbone again
            if self.multi_scale:
                video_short = video[:, ::2, :, :, :]  # [B, T//2, H, W, C]
                T_short = int(video_short.shape[1])
                tokens_short = self.backbone(video_short)  # [B, T_s*S, D]
            else:
                tokens_short = None
                T_short = None

        if cache_backbone_tokens:
            self._backbone_tokens = tokens_full

        self._frame_output = None
        frame_device = next(self.frame_head.parameters()).device
        tokens_in = tokens_full.to(frame_device) if tokens_full.device != frame_device else tokens_full
        tokens_short_in = (
            tokens_short.to(frame_device)
            if tokens_short is not None and tokens_short.device != frame_device
            else tokens_short
        )
        frame_out = self.frame_head(
            tokens_in, num_frames=T,
            tokens_short=tokens_short_in,
            num_frames_short=T_short,
            return_attn_weights=return_attn_weights,
        )

        frame_logits = frame_out[0]
        frame_clip_logits = frame_out[1]
        temporal_weights = frame_out[2]
        frame_logits_pooled = frame_out[3]
        boundary_logits = frame_out[4]
        frame_embeddings = frame_out[5]
        frame_embeddings_combined = frame_out[6] if len(frame_out) > 6 else None
        attn_weights = frame_out[7] if len(frame_out) > 7 else None

        if return_frame_logits:
            self._frame_output = (
                frame_logits,          # 0  [B, T, C]
                frame_clip_logits,     # 1  [B, C]
                temporal_weights,      # 2  [B, T]
                frame_logits_pooled,   # 3  [B, T_pooled, C]
                int(self.temporal_pool_frames),  # 4
                boundary_logits,       # 5  [B, T, 1]
                frame_embeddings,      # 6  [B, T, proj_dim]  (attention-pooled)
                frame_embeddings_combined,  # 7  [B, T, 2*proj_dim] (attn || raw_mean)
                attn_weights,          # 8  [B, T, num_heads, S] or None
            )

        head_output = frame_clip_logits

        if return_localization and self.use_localization and self.localization_head is not None:
            loc_device = next(self.localization_head.parameters()).device
            loc_out = self.localization_head(
                tokens_full.to(loc_device) if tokens_full.device != loc_device else tokens_full,
                num_frames=T,
                fixed_box_wh=localization_box_wh,
            )
            return head_output, loc_out

        return head_output

    def save_head(self, path: str, metadata: Optional[dict] = None):
        """Save frame head (and localization head) parameters."""
        payload = {
            "frame_head_state_dict": self.frame_head.state_dict(),
            "use_localization": self.use_localization,
            "use_frame_head": True,
            "multi_scale": self.multi_scale,
            "use_temporal_decoder": self.use_temporal_decoder,
        }
        if self.use_localization and self.localization_head is not None:
            payload["localization_state_dict"] = self.localization_head.state_dict()
        torch.save(payload, path)

        if metadata:
            import json
            meta_path = path + ".meta.json"
            try:
                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump(metadata, f, indent=2)
            except Exception as exc:
                logger.warning("Failed to write metadata to %s: %s", meta_path, exc)

    def load_head(self, path: str):
        """Load head parameters."""
        state = torch.load(path, map_location='cpu')
        if not isinstance(state, dict):
            return

        if "frame_head_state_dict" in state:
            self.frame_head.load_state_dict(
                state["frame_head_state_dict"], strict=False,
            )
        if (
            self.use_localization
            and self.localization_head is not None
            and isinstance(state.get("localization_state_dict"), dict)
        ):
            self.localization_head.load_state_dict(
                state["localization_state_dict"], strict=False,
            )


# Loss functions.

def _giou(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Generalized IoU between xyxy boxes. Returns [N] in [-1, 1]."""
    eps = 1e-7
    ix1 = torch.maximum(pred[:, 0], target[:, 0])
    iy1 = torch.maximum(pred[:, 1], target[:, 1])
    ix2 = torch.minimum(pred[:, 2], target[:, 2])
    iy2 = torch.minimum(pred[:, 3], target[:, 3])
    inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)

    area_p = (pred[:, 2] - pred[:, 0]).clamp(min=0) * (pred[:, 3] - pred[:, 1]).clamp(min=0)
    area_t = (target[:, 2] - target[:, 0]).clamp(min=0) * (target[:, 3] - target[:, 1]).clamp(min=0)
    union = area_p + area_t - inter
    iou = inter / (union + eps)

    ex1 = torch.minimum(pred[:, 0], target[:, 0])
    ey1 = torch.minimum(pred[:, 1], target[:, 1])
    ex2 = torch.maximum(pred[:, 2], target[:, 2])
    ey2 = torch.maximum(pred[:, 3], target[:, 3])
    area_enclosing = (ex2 - ex1).clamp(min=0) * (ey2 - ey1).clamp(min=0)

    return iou - (area_enclosing - union) / (area_enclosing + eps)


def localization_bbox_loss(
    pred_bboxes: torch.Tensor,
    target_bboxes: torch.Tensor,
    valid_mask: torch.Tensor,
    smooth_l1_weight: float = 0.5,
    giou_weight: float = 0.5,
    temporal_smoothness_weight: float = 0.0,
) -> torch.Tensor:
    """Combined Smooth-L1 + GIoU bbox loss over valid samples."""
    if pred_bboxes is None or target_bboxes is None or valid_mask is None:
        return torch.tensor(0.0, device=pred_bboxes.device if pred_bboxes is not None else "cpu", requires_grad=True)

    valid = valid_mask > 0.5
    if not valid.any():
        return torch.tensor(0.0, device=pred_bboxes.device, requires_grad=True)

    pred_first = pred_bboxes[:, 0, :] if pred_bboxes.dim() == 3 else pred_bboxes
    pred = pred_first[valid]
    tgt = target_bboxes[valid]

    loss_l1 = F.smooth_l1_loss(pred, tgt, reduction="mean")
    loss_giou = (1.0 - _giou(pred, tgt)).mean()
    loss = smooth_l1_weight * loss_l1 + giou_weight * loss_giou

    if pred_bboxes.dim() == 3 and pred_bboxes.size(1) > 1:
        pred_valid_seq = pred_bboxes[valid]
        if pred_valid_seq.numel() > 0:
            smooth = F.smooth_l1_loss(
                pred_valid_seq[:, 1:, :], pred_valid_seq[:, :-1, :],
                reduction="mean",
            )
            loss = loss + temporal_smoothness_weight * smooth

    return loss


def objectness_spatial_contrastive_loss(
    projected_tokens: torch.Tensor,
    spatial_masks: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Per-sample spatial contrastive loss for localization training."""
    device = projected_tokens.device
    B, S, D = projected_tokens.shape

    if spatial_masks.size(1) != S:
        return torch.tensor(0.0, device=device, requires_grad=True)

    n_inside = spatial_masks.sum(dim=1)
    n_outside = S - n_inside
    valid = (n_inside >= 2) & (n_outside >= 1)
    if not valid.any():
        return torch.tensor(0.0, device=device, requires_grad=True)

    tokens_sub = F.normalize(projected_tokens[valid], p=2, dim=-1)
    masks_sub = spatial_masks[valid]

    total_loss = torch.tensor(0.0, device=device)
    count = 0

    for i in range(tokens_sub.size(0)):
        tok = tokens_sub[i]
        mask = masks_sub[i]
        inside_idx = (mask > 0.5).nonzero(as_tuple=True)[0]
        outside_idx = (mask <= 0.5).nonzero(as_tuple=True)[0]
        K = inside_idx.size(0)
        if K < 2 or outside_idx.size(0) < 1:
            continue

        inside_tok = tok[inside_idx]
        outside_tok = tok[outside_idx]
        sim_pos = torch.matmul(inside_tok, inside_tok.T) / temperature
        sim_neg = torch.matmul(inside_tok, outside_tok.T) / temperature
        sim_pos = sim_pos.masked_fill(
            torch.eye(K, device=device, dtype=torch.bool), float("-inf"),
        )
        log_numer = torch.logsumexp(sim_pos, dim=1)
        all_sim = torch.cat([sim_pos, sim_neg], dim=1)
        log_denom = torch.logsumexp(all_sim, dim=1)
        total_loss = total_loss + (-(log_numer - log_denom)).mean()
        count += 1

    if count == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    return total_loss / count


def objectness_mask_loss(
    objectness_logits: torch.Tensor,
    spatial_masks: torch.Tensor,
    min_pos_tokens: int = 2,
) -> torch.Tensor:
    """BCE logits loss for localization objectness supervision."""
    device = objectness_logits.device
    if spatial_masks.shape != objectness_logits.shape:
        return torch.tensor(0.0, device=device, requires_grad=True)

    pos_count = spatial_masks.sum(dim=1)
    valid = (pos_count >= float(min_pos_tokens)) & (pos_count < float(spatial_masks.size(1)))
    if not valid.any():
        return torch.tensor(0.0, device=device, requires_grad=True)

    logits = objectness_logits[valid]
    targets = spatial_masks[valid].float()
    pos = targets.sum()
    neg = torch.tensor(float(targets.numel()), device=device) - pos
    pos_weight = (neg / (pos + 1e-6)).clamp(min=1.0, max=20.0)
    return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)


def gaussian_focal_loss(
    pred_logits: torch.Tensor,
    target_heatmap: torch.Tensor,
    alpha: float = 2.0,
    beta: float = 4.0,
) -> torch.Tensor:
    """CenterNet-style Gaussian focal loss for heatmap supervision."""
    pred = torch.sigmoid(pred_logits)
    pos_mask = target_heatmap.eq(1.0)
    neg_mask = ~pos_mask

    pos_loss = -((1 - pred).pow(alpha) * torch.log(pred.clamp(min=1e-6))) * pos_mask
    neg_loss = -(
        (1 - target_heatmap).pow(beta) * pred.pow(alpha)
        * torch.log((1 - pred).clamp(min=1e-6))
    ) * neg_mask

    num_pos = pos_mask.float().sum().clamp(min=1.0)
    return (pos_loss.sum() + neg_loss.sum()) / num_pos


def center_heatmap_loss(
    objectness_logits: torch.Tensor,
    target_bboxes: torch.Tensor,
    valid_mask: torch.Tensor,
    sigma_in_patches: float = 1.5,
) -> torch.Tensor:
    """Gaussian center heatmap supervision for localization objectness."""
    device = objectness_logits.device
    B, S = objectness_logits.shape
    if target_bboxes is None or valid_mask is None:
        return torch.tensor(0.0, device=device, requires_grad=True)
    if target_bboxes.dim() != 2 or target_bboxes.size(0) != B or target_bboxes.size(1) != 4:
        return torch.tensor(0.0, device=device, requires_grad=True)

    valid = valid_mask > 0.5
    if not valid.any():
        return torch.tensor(0.0, device=device, requires_grad=True)

    g = int(round(S ** 0.5))
    if g * g != S:
        return torch.tensor(0.0, device=device, requires_grad=True)

    ys = (torch.arange(g, device=device, dtype=objectness_logits.dtype) + 0.5) / float(g)
    xs = (torch.arange(g, device=device, dtype=objectness_logits.dtype) + 0.5) / float(g)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")

    tgt = target_bboxes[valid]
    cx = (0.5 * (tgt[:, 0] + tgt[:, 2])).view(-1, 1, 1)
    cy = (0.5 * (tgt[:, 1] + tgt[:, 3])).view(-1, 1, 1)

    sigma = max(1e-4, float(sigma_in_patches) / float(g))
    d2 = (xx.view(1, g, g) - cx).pow(2) + (yy.view(1, g, g) - cy).pow(2)
    target_heat = torch.exp(-d2 / (2.0 * sigma * sigma)).clamp(0.0, 1.0)

    Bv = target_heat.size(0)
    flat = target_heat.view(Bv, -1)
    peak_idx = flat.argmax(dim=1, keepdim=True)
    flat.scatter_(1, peak_idx, 1.0)
    target_heat = flat.view(Bv, g, g)

    pred_logits_v = objectness_logits[valid].view(-1, g, g)
    return gaussian_focal_loss(pred_logits_v, target_heat)


def direct_center_loss(
    pred_bboxes: torch.Tensor,
    target_bboxes: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Center-to-center SmoothL1 loss for localization."""
    if pred_bboxes is None or target_bboxes is None or valid_mask is None:
        dev = pred_bboxes.device if pred_bboxes is not None else "cpu"
        return torch.tensor(0.0, device=dev, requires_grad=True)

    valid = valid_mask > 0.5
    if not valid.any():
        return torch.tensor(0.0, device=pred_bboxes.device, requires_grad=True)

    pred_first = pred_bboxes[:, 0, :] if pred_bboxes.dim() == 3 else pred_bboxes
    pred = pred_first[valid]
    tgt = target_bboxes[valid]

    pred_center = torch.stack([
        0.5 * (pred[:, 0] + pred[:, 2]),
        0.5 * (pred[:, 1] + pred[:, 3]),
    ], dim=1)
    tgt_center = torch.stack([
        0.5 * (tgt[:, 0] + tgt[:, 2]),
        0.5 * (tgt[:, 1] + tgt[:, 3]),
    ], dim=1)

    return F.smooth_l1_loss(pred_center, tgt_center, reduction="mean")


def frame_classification_loss(
    frame_logits: torch.Tensor,
    frame_labels: torch.Tensor,
) -> torch.Tensor:
    """Per-frame CE loss, ignoring frames with label -1."""
    B, T, C = frame_logits.shape
    device = frame_logits.device
    logits_flat = frame_logits.reshape(B * T, C)
    labels_flat = frame_labels.reshape(B * T)
    valid = labels_flat >= 0
    if not valid.any():
        return torch.tensor(0.0, device=device, requires_grad=True)
    return F.cross_entropy(logits_flat[valid], labels_flat[valid])


def boundary_detection_loss(
    boundary_logits: torch.Tensor,
    boundary_labels: torch.Tensor,
) -> torch.Tensor:
    """BCE loss for boundary/change-point detection.

    Args:
        boundary_logits: [B, T, 1] raw logits
        boundary_labels: [B, T] binary (1=transition, 0=no, -1=ignore)
    """
    B, T, _ = boundary_logits.shape
    device = boundary_logits.device
    logits_flat = boundary_logits.squeeze(-1)  # [B, T]
    valid = boundary_labels >= 0
    if not valid.any():
        return torch.tensor(0.0, device=device, requires_grad=True)

    n_pos = (boundary_labels[valid] > 0.5).float().sum().clamp(min=1.0)
    n_neg = (boundary_labels[valid] <= 0.5).float().sum().clamp(min=1.0)
    pos_weight = (n_neg / n_pos).clamp(max=20.0)

    raw = F.binary_cross_entropy_with_logits(
        logits_flat[valid], boundary_labels[valid].float(),
        pos_weight=pos_weight, reduction='none',
    )

    return raw.mean()


def temporal_smoothness_loss(
    frame_logits: torch.Tensor,
    frame_labels: torch.Tensor,
) -> torch.Tensor:
    """L2 smoothness regularizer on consecutive frame logits."""
    B, T, C = frame_logits.shape
    if T < 2:
        return torch.tensor(0.0, device=frame_logits.device, requires_grad=True)

    valid = frame_labels >= 0
    both_valid = valid[:, :-1] & valid[:, 1:]
    if not both_valid.any():
        return torch.tensor(0.0, device=frame_logits.device, requires_grad=True)

    diff = (frame_logits[:, 1:, :] - frame_logits[:, :-1, :]) ** 2
    return diff.mean(dim=-1)[both_valid].mean()
