import torch
import torch.nn as nn
import torchvision.transforms.functional as F
from typing import Optional


class ClipAugment(nn.Module):
    """Applies identical augmentation to every frame; no cropping to avoid cutting off tracked subjects."""
    
    def __init__(
        self,
        use_horizontal_flip: bool = True,
        use_vertical_flip: bool = False,
        use_color_jitter: bool = True,
        use_gaussian_blur: bool = False,
        use_random_noise: bool = False,
        use_small_rotation: bool = False,
        use_speed_perturb: bool = False,
        use_random_shapes: bool = False,
        use_grayscale: bool = False,
        use_lighting_robustness: bool = False,
        color_jitter_brightness: float = 0.2,
        color_jitter_contrast: float = 0.2,
        color_jitter_saturation: float = 0.2,
        color_jitter_hue: float = 0.1,
        gaussian_blur_sigma: tuple[float, float] = (0.1, 0.5),
        noise_std: float = 0.01,
        rotation_degrees: float = 2.0,
        speed_range: tuple[float, float] = (0.7, 1.3),
        random_shapes_max: int = 3,
        random_shapes_max_size: float = 0.15,
        grayscale_prob: float = 0.5,
        gamma_range: tuple[float, float] = (0.75, 1.35),
        channel_gain_range: tuple[float, float] = (0.85, 1.15),
    ):
        super().__init__()
        self.use_horizontal_flip = use_horizontal_flip
        self.use_vertical_flip = use_vertical_flip
        self.use_color_jitter = use_color_jitter
        self.use_gaussian_blur = use_gaussian_blur
        self.use_random_noise = use_random_noise
        self.use_small_rotation = use_small_rotation
        self.use_speed_perturb = use_speed_perturb
        self.use_random_shapes = use_random_shapes
        self.use_grayscale = use_grayscale
        self.use_lighting_robustness = use_lighting_robustness
        
        self.color_jitter_brightness = color_jitter_brightness
        self.color_jitter_contrast = color_jitter_contrast
        self.color_jitter_saturation = color_jitter_saturation
        self.color_jitter_hue = color_jitter_hue
        self.gaussian_blur_sigma = gaussian_blur_sigma
        self.noise_std = noise_std
        self.rotation_degrees = rotation_degrees
        self.speed_range = speed_range
        self.random_shapes_max = max(1, random_shapes_max)
        self.random_shapes_max_size = random_shapes_max_size
        self.grayscale_prob = grayscale_prob
        self.gamma_range = gamma_range
        self.channel_gain_range = channel_gain_range
    
    def _sample_params(self) -> dict:
        """Sample one set of augmentation params shared by all frames."""
        hflip = self.use_horizontal_flip and (torch.rand(1).item() < 0.5)
        vflip = self.use_vertical_flip and (torch.rand(1).item() < 0.5)

        if self.use_color_jitter:
            brightness_factor = torch.empty(1).uniform_(
                1 - self.color_jitter_brightness,
                1 + self.color_jitter_brightness
            ).item() if self.color_jitter_brightness > 0 else 1.0
            
            contrast_factor = torch.empty(1).uniform_(
                1 - self.color_jitter_contrast,
                1 + self.color_jitter_contrast
            ).item() if self.color_jitter_contrast > 0 else 1.0
            
            saturation_factor = torch.empty(1).uniform_(
                1 - self.color_jitter_saturation,
                1 + self.color_jitter_saturation
            ).item() if self.color_jitter_saturation > 0 else 1.0
            
            hue_factor = torch.empty(1).uniform_(
                -self.color_jitter_hue,
                self.color_jitter_hue
            ).item() if self.color_jitter_hue > 0 else 0.0
        else:
            brightness_factor = contrast_factor = saturation_factor = hue_factor = None
        
        if self.use_gaussian_blur:
            blur_sigma = torch.empty(1).uniform_(
                self.gaussian_blur_sigma[0],
                self.gaussian_blur_sigma[1]
            ).item()
        else:
            blur_sigma = None
        
        if self.use_small_rotation:
            rotation_angle = torch.empty(1).uniform_(
                -self.rotation_degrees,
                self.rotation_degrees
            ).item()
        else:
            rotation_angle = None
        
        if self.use_random_noise:
            noise_std = self.noise_std
        else:
            noise_std = None

        gamma_factor = None
        channel_gains = None
        if self.use_lighting_robustness:
            gamma_factor = torch.empty(1).uniform_(
                self.gamma_range[0],
                self.gamma_range[1]
            ).item()
            channel_gains = torch.empty(3).uniform_(
                self.channel_gain_range[0],
                self.channel_gain_range[1]
            ).tolist()

        # Speed perturbation: sample a speed factor once per clip
        speed_factor = None
        if self.use_speed_perturb:
            speed_factor = torch.empty(1).uniform_(
                self.speed_range[0], self.speed_range[1]
            ).item()

        # Random shapes: sample positions, sizes, colors, types once per clip
        shapes = None
        if self.use_random_shapes:
            n_shapes = torch.randint(1, self.random_shapes_max + 1, (1,)).item()
            shapes = []
            for _ in range(n_shapes):
                shape_type = torch.randint(0, 3, (1,)).item()  # 0=rect, 1=ellipse, 2=triangle
                cx = torch.rand(1).item()
                cy = torch.rand(1).item()
                sw = torch.empty(1).uniform_(0.03, self.random_shapes_max_size).item()
                sh = torch.empty(1).uniform_(0.03, self.random_shapes_max_size).item()
                color = torch.rand(3).tolist()
                shapes.append({
                    "type": shape_type, "cx": cx, "cy": cy,
                    "sw": sw, "sh": sh, "color": color,
                })

        do_grayscale = False
        if self.use_grayscale:
            do_grayscale = torch.rand(1).item() < self.grayscale_prob

        return {
            "hflip": bool(hflip),
            "vflip": bool(vflip),
            "brightness_factor": brightness_factor,
            "contrast_factor": contrast_factor,
            "saturation_factor": saturation_factor,
            "hue_factor": hue_factor,
            "blur_sigma": blur_sigma,
            "rotation_angle": rotation_angle,
            "noise_std": noise_std,
            "gamma_factor": gamma_factor,
            "channel_gains": channel_gains,
            "speed_factor": speed_factor,
            "shapes": shapes,
            "grayscale": do_grayscale,
        }

    @staticmethod
    def _resample_temporal(clip: torch.Tensor, speed_factor: float) -> torch.Tensor:
        """Resample clip frames to simulate speed change. Output has same T."""
        T = clip.shape[0]
        if T <= 1 or abs(speed_factor - 1.0) < 0.01:
            return clip
        # At speed_factor > 1 we want to cover more source frames (speed up),
        # so we sample from a wider window (indices can exceed T-1 → clamp).
        # At speed_factor < 1 we cover fewer source frames (slow down),
        # so indices cluster in the center.
        src_indices = torch.linspace(0, (T - 1) * speed_factor, T)
        src_indices = src_indices.clamp(0, T - 1)
        idx = src_indices.round().long()
        return clip[idx]

    @staticmethod
    def _draw_shapes_on_frame(frame: torch.Tensor, shapes: list, H: int, W: int) -> torch.Tensor:
        """Draw pre-sampled shapes onto a single frame [C, H, W]."""
        frame = frame.clone()
        for s in shapes:
            cx_px = int(s["cx"] * W)
            cy_px = int(s["cy"] * H)
            half_w = max(1, int(s["sw"] * W / 2))
            half_h = max(1, int(s["sh"] * H / 2))
            color = s["color"]

            x1 = max(0, cx_px - half_w)
            x2 = min(W, cx_px + half_w)
            y1 = max(0, cy_px - half_h)
            y2 = min(H, cy_px + half_h)
            if x2 <= x1 or y2 <= y1:
                continue

            stype = s["type"]
            if stype == 0:
                for c_i in range(min(3, frame.shape[0])):
                    frame[c_i, y1:y2, x1:x2] = color[c_i]
            elif stype == 1:
                yy = torch.arange(y1, y2, device=frame.device).float()
                xx = torch.arange(x1, x2, device=frame.device).float()
                gy, gx = torch.meshgrid(yy, xx, indexing="ij")
                ey = (gy - cy_px) / max(half_h, 1)
                ex = (gx - cx_px) / max(half_w, 1)
                mask = (ex ** 2 + ey ** 2) <= 1.0
                for c_i in range(min(3, frame.shape[0])):
                    region = frame[c_i, y1:y2, x1:x2]
                    region[mask] = color[c_i]
            elif stype == 2:
                # Triangle pointing up: apex at top center, base at bottom
                yy = torch.arange(y1, y2, device=frame.device).float()
                xx = torch.arange(x1, x2, device=frame.device).float()
                gy, gx = torch.meshgrid(yy, xx, indexing="ij")
                ny = (gy - y1) / max(y2 - y1 - 1, 1)
                nx = (gx - x1) / max(x2 - x1 - 1, 1)
                mask = (nx >= 0.5 - 0.5 * ny) & (nx <= 0.5 + 0.5 * ny)
                for c_i in range(min(3, frame.shape[0])):
                    region = frame[c_i, y1:y2, x1:x2]
                    region[mask] = color[c_i]
        return frame

    def _apply_with_params(self, clip: torch.Tensor, params: dict) -> torch.Tensor:
        T, C, H, W = clip.shape

        # Speed perturbation (resamples frames, applied before per-frame ops)
        speed_factor = params.get("speed_factor", None)
        if speed_factor is not None:
            clip = self._resample_temporal(clip, speed_factor)

        hflip = bool(params.get("hflip", params.get("flip", False)))
        vflip = bool(params.get("vflip", False))
        brightness_factor = params.get("brightness_factor", None)
        contrast_factor = params.get("contrast_factor", None)
        saturation_factor = params.get("saturation_factor", None)
        hue_factor = params.get("hue_factor", None)
        blur_sigma = params.get("blur_sigma", None)
        rotation_angle = params.get("rotation_angle", None)
        noise_std = params.get("noise_std", None)
        gamma_factor = params.get("gamma_factor", None)
        channel_gains = params.get("channel_gains", None)
        shapes = params.get("shapes", None)
        do_grayscale = params.get("grayscale", False)

        augmented_frames = []
        for t in range(T):
            frame = clip[t]

            if hflip:
                frame = F.hflip(frame)
            if vflip:
                frame = F.vflip(frame)

            if do_grayscale:
                frame = F.rgb_to_grayscale(frame, num_output_channels=C)

            if brightness_factor is not None:
                frame = F.adjust_brightness(frame, brightness_factor)
                frame = F.adjust_contrast(frame, contrast_factor)
                frame = F.adjust_saturation(frame, saturation_factor)
                frame = F.adjust_hue(frame, hue_factor)

            # Gamma correction + per-channel gain shifts for lighting robustness
            if gamma_factor is not None:
                frame = torch.clamp(frame, 0.0, 1.0)
                frame = torch.pow(frame + 1e-6, gamma_factor)
                if channel_gains is not None and C >= 3:
                    gains = torch.tensor(channel_gains[:3], device=frame.device, dtype=frame.dtype).view(3, 1, 1)
                    frame[:3] = frame[:3] * gains
                frame = torch.clamp(frame, 0.0, 1.0)
            
            if rotation_angle is not None and abs(rotation_angle) > 0.01:
                frame = F.rotate(frame, rotation_angle, interpolation=F.InterpolationMode.BILINEAR, fill=0.0)

            if blur_sigma is not None and blur_sigma > 0.01:
                kernel_size = int(2 * int(4 * blur_sigma + 0.5) + 1)
                if kernel_size >= 3:
                    frame = F.gaussian_blur(frame, kernel_size=[kernel_size, kernel_size], sigma=[blur_sigma, blur_sigma])
            
            # Random shapes (same position/color for every frame in the clip)
            if shapes:
                frame = self._draw_shapes_on_frame(frame, shapes, H, W)

            if noise_std is not None and noise_std > 0:
                noise = torch.randn_like(frame) * noise_std
                frame = torch.clamp(frame + noise, 0.0, 1.0)
            
            augmented_frames.append(frame)

        return torch.stack(augmented_frames)

    def augment_with_params(self, clip: torch.Tensor):
        """Return augmented clip and parameter dict for synchronizing spatial label transforms (e.g., bboxes, masks)."""
        params = self._sample_params()
        return self._apply_with_params(clip, params), params

    def forward(self, clip: torch.Tensor) -> torch.Tensor:
        """clip: [T, C, H, W] float tensor in [0, 1]."""
        augmented_clip, _ = self.augment_with_params(clip)
        return augmented_clip
    
    def __repr__(self):
        parts = [
            f"hflip={self.use_horizontal_flip}",
            f"vflip={self.use_vertical_flip}",
            f"color_jitter={self.use_color_jitter}",
            f"gaussian_blur={self.use_gaussian_blur}",
            f"random_noise={self.use_random_noise}",
            f"small_rotation={self.use_small_rotation}",
            f"speed_perturb={self.use_speed_perturb}",
            f"random_shapes={self.use_random_shapes}",
            f"grayscale={self.use_grayscale}",
            f"lighting_robustness={self.use_lighting_robustness}",
        ]
        return f"ClipAugment({', '.join(parts)})"
