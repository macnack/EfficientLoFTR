from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import kornia.augmentation as K
import torch
import torch.nn as nn


@dataclass(frozen=True)
class AugmentationConfig:
    enabled: bool = True
    apply_same_on_pair: bool = True
    image_keys: Sequence[str] = ("im_A", "im_B")
    color_jitter: Optional[Mapping[str, float]] = None
    daytime_simulation: Optional[Mapping[str, float]] = None
    gaussian_blur: Optional[Mapping[str, float]] = None
    gaussian_noise: Optional[Mapping[str, float]] = None
    random_erasing: Optional[Mapping[str, float]] = None

    @classmethod
    def from_yaml(cls, path: str) -> "AugmentationConfig":
        try:
            import yaml
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "PyYAML is required to load augmentation config files. "
                "Install it with `pip install pyyaml`."
            ) from exc

        with open(path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        if not isinstance(data, Mapping):
            raise ValueError("AugmentationConfig YAML must contain a mapping at the top level.")
        return cls(**data)


class DaytimeSimulation(nn.Module):
    """Simulate day/night lighting via smooth brightness curves and temperature shifts."""

    def __init__(
        self,
        *,
        p: float = 0.5,
        brightness_range: Tuple[float, float] = (0.6, 1.4),
        gamma_range: Tuple[float, float] = (0.8, 1.2),
        curve_strength: float = 0.35,
        temperature_shift: float = 0.05,
        same_on_batch: bool = True,
    ) -> None:
        super().__init__()
        self.p = p
        self.brightness_range = brightness_range
        self.gamma_range = gamma_range
        self.curve_strength = curve_strength
        self.temperature_shift = temperature_shift
        self.same_on_batch = same_on_batch

    def _sample(self, batch: int, device: torch.device, dtype: torch.dtype) -> Dict[str, torch.Tensor]:
        if self.same_on_batch:
            brightness = torch.empty(1, device=device, dtype=dtype).uniform_(
                *self.brightness_range
            )
            gamma = torch.empty(1, device=device, dtype=dtype).uniform_(*self.gamma_range)
            temperature = torch.empty(1, device=device, dtype=dtype).uniform_(
                -self.temperature_shift, self.temperature_shift
            )
        else:
            brightness = torch.empty(batch, device=device, dtype=dtype).uniform_(
                *self.brightness_range
            )
            gamma = torch.empty(batch, device=device, dtype=dtype).uniform_(*self.gamma_range)
            temperature = torch.empty(batch, device=device, dtype=dtype).uniform_(
                -self.temperature_shift, self.temperature_shift
            )
        return {
            "brightness": brightness,
            "gamma": gamma,
            "temperature": temperature,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.p <= 0:
            return x
        if x.dim() != 4:
            raise ValueError("DaytimeSimulation expects a 4D tensor [B, C, H, W].")
        batch, channels, _, _ = x.shape
        if channels != 3:
            return x

        if self.p < 1.0:
            apply_mask = torch.rand(
                batch, device=x.device, dtype=x.dtype
            ) < self.p
        else:
            apply_mask = torch.ones(batch, device=x.device, dtype=x.dtype).bool()

        params = self._sample(batch, x.device, x.dtype)
        brightness = params["brightness"].view(-1, 1, 1, 1)
        gamma = params["gamma"].view(-1, 1, 1, 1)
        temperature = params["temperature"].view(-1, 1, 1, 1)

        # Smooth nonlinear curve (monotonic, invertible for gamma > 0)
        curved = x.clamp(0.0, 1.0).pow(gamma)
        curved = curved + self.curve_strength * (curved - curved.pow(2))
        curved = curved * brightness

        # Temperature shift: warm (increase red), cool (increase blue)
        temp_scale = torch.cat(
            [
                1.0 + temperature,
                torch.ones_like(temperature),
                1.0 - temperature,
            ],
            dim=1,
        ).view(-1, 3, 1, 1)
        curved = curved * temp_scale
        curved = curved.clamp(0.0, 1.0)

        if apply_mask.all():
            return curved

        out = x.clone()
        out[apply_mask] = curved[apply_mask]
        return out


class AugmentationPipeline:
    """Configurable augmentation pipeline for paired satellite images."""

    def __init__(
        self,
        config: Optional[AugmentationConfig] = None,
        *,
        enabled: Optional[bool] = None,
        apply_same_on_pair: Optional[bool] = None,
        image_keys: Optional[Sequence[str]] = None,
        color_jitter: Optional[Mapping[str, float]] = None,
        daytime_simulation: Optional[Mapping[str, float]] = None,
        gaussian_blur: Optional[Mapping[str, float]] = None,
        gaussian_noise: Optional[Mapping[str, float]] = None,
        random_erasing: Optional[Mapping[str, float]] = None,
    ) -> None:
        base = config or AugmentationConfig()
        self.enabled = base.enabled if enabled is None else enabled
        self.apply_same_on_pair = (
            base.apply_same_on_pair if apply_same_on_pair is None else apply_same_on_pair
        )
        self.image_keys = base.image_keys if image_keys is None else image_keys
        self._transforms = self._build_transforms(
            color_jitter=color_jitter if color_jitter is not None else base.color_jitter,
            daytime_simulation=(
                daytime_simulation
                if daytime_simulation is not None
                else base.daytime_simulation
            ),
            gaussian_blur=gaussian_blur if gaussian_blur is not None else base.gaussian_blur,
            gaussian_noise=gaussian_noise if gaussian_noise is not None else base.gaussian_noise,
            random_erasing=random_erasing if random_erasing is not None else base.random_erasing,
        )
        self.pipeline = (
            K.AugmentationSequential(*self._transforms, data_keys=["input"])
            if self._transforms
            else None
        )

    def _resolve_config(
        self,
        config: Optional[Mapping[str, float]],
        defaults: Mapping[str, float],
    ) -> Optional[Mapping[str, float]]:
        if config is None:
            return defaults
        if config is False:
            return None
        merged = dict(defaults)
        merged.update(config)
        return merged

    def _build_transforms(
        self,
        *,
        color_jitter: Optional[Mapping[str, float]],
        daytime_simulation: Optional[Mapping[str, float]],
        gaussian_blur: Optional[Mapping[str, float]],
        gaussian_noise: Optional[Mapping[str, float]],
        random_erasing: Optional[Mapping[str, float]],
    ) -> Sequence[nn.Module]:
        transforms: list[nn.Module] = []

        color_cfg = self._resolve_config(
            color_jitter,
            {
                "brightness": 0.2,
                "contrast": 0.2,
                "saturation": 0.2,
                "hue": 0.1,
                "p": 0.8,
            },
        )
        if color_cfg is not None:
            transforms.append(
                K.ColorJitter(
                    brightness=color_cfg["brightness"],
                    contrast=color_cfg["contrast"],
                    saturation=color_cfg["saturation"],
                    hue=color_cfg["hue"],
                    p=color_cfg.get("p", 1.0),
                    same_on_batch=self.apply_same_on_pair,
                )
            )

        daytime_cfg = self._resolve_config(
            daytime_simulation,
            {
                "p": 0.5,
                "brightness_min": 0.6,
                "brightness_max": 1.4,
                "gamma_min": 0.8,
                "gamma_max": 1.2,
                "curve_strength": 0.35,
                "temperature_shift": 0.05,
            },
        )
        if daytime_cfg is not None:
            transforms.append(
                DaytimeSimulation(
                    p=daytime_cfg.get("p", 0.5),
                    brightness_range=(
                        daytime_cfg["brightness_min"],
                        daytime_cfg["brightness_max"],
                    ),
                    gamma_range=(
                        daytime_cfg["gamma_min"],
                        daytime_cfg["gamma_max"],
                    ),
                    curve_strength=daytime_cfg["curve_strength"],
                    temperature_shift=daytime_cfg["temperature_shift"],
                    same_on_batch=self.apply_same_on_pair,
                )
            )

        blur_cfg = self._resolve_config(
            gaussian_blur,
            {
                "kernel_size": 3,
                "sigma": 1.0,
                "p": 0.3,
            },
        )
        if blur_cfg is not None:
            transforms.append(
                K.RandomGaussianBlur(
                    kernel_size=(int(blur_cfg["kernel_size"]), int(blur_cfg["kernel_size"])),
                    sigma=(float(blur_cfg["sigma"]), float(blur_cfg["sigma"])),
                    p=blur_cfg.get("p", 1.0),
                    same_on_batch=self.apply_same_on_pair,
                )
            )

        noise_cfg = self._resolve_config(
            gaussian_noise,
            {
                "mean": 0.0,
                "std": 0.02,
                "p": 0.3,
            },
        )
        if noise_cfg is not None:
            transforms.append(
                K.RandomGaussianNoise(
                    mean=noise_cfg["mean"],
                    std=noise_cfg["std"],
                    p=noise_cfg.get("p", 1.0),
                    same_on_batch=self.apply_same_on_pair,
                )
            )

        erasing_cfg = self._resolve_config(
            random_erasing,
            {
                "scale": 0.05,
                "p": 0.2,
            },
        )
        if erasing_cfg is not None:
            transforms.append(
                K.RandomErasing(
                    scale=(0.02, erasing_cfg["scale"]),
                    p=erasing_cfg.get("p", 1.0),
                    same_on_batch=self.apply_same_on_pair,
                )
            )

        return transforms

    def _stack_images(self, images: Iterable[torch.Tensor]) -> torch.Tensor:
        batch = []
        for image in images:
            if image.dim() == 3:
                batch.append(image.unsqueeze(0))
            elif image.dim() == 4:
                batch.append(image)
            else:
                raise ValueError(
                    "AugmentationPipeline expects image tensors with 3 or 4 dimensions."
                )
        return torch.cat(batch, dim=0)

    def _image_shape(self, image: torch.Tensor) -> Tuple[int, int, int]:
        if image.dim() == 3:
            return (image.shape[0], image.shape[1], image.shape[2])
        if image.dim() == 4:
            return (image.shape[1], image.shape[2], image.shape[3])
        raise ValueError("AugmentationPipeline expects image tensors with 3 or 4 dimensions.")

    def _can_stack(self, images: Sequence[torch.Tensor]) -> bool:
        if not images:
            return False
        reference = self._image_shape(images[0])
        return all(self._image_shape(image) == reference for image in images)

    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if not self.enabled or self.pipeline is None:
            return sample

        keys = [key for key in self.image_keys if key in sample]
        if not keys:
            return sample

        images = [sample[key] for key in keys]
        if self.apply_same_on_pair and self._can_stack(images):
            stacked = self._stack_images(images)
            augmented = self.pipeline(stacked)
            offset = 0
            for key, image in zip(keys, images):
                count = image.shape[0] if image.dim() == 4 else 1
                sample[key] = augmented[offset : offset + count].squeeze(0)
                offset += count
        else:
            for key in keys:
                image = sample[key]
                if image.dim() == 3:
                    image = image.unsqueeze(0)
                sample[key] = self.pipeline(image).squeeze(0)

        return sample
