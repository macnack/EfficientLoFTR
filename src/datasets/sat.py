import torch
import numpy as np
import cv2
import pyvips
import random
from pathlib import Path
from torch.utils.data import Dataset
import torch.nn.functional as F
import kornia
import torch.nn as nn
from typing import Tuple, Dict, Optional, List, Union
from torchvision.transforms import Normalize

from src.datasets.augmentation import AugmentationPipeline, AugmentationConfig
from src.utils.dataset import get_resized_wh, get_divisible_wh, pad_bottom_right

_RGB_TO_GRAY_WEIGHTS = torch.tensor([0.2989, 0.5870, 0.1140]).view(3, 1, 1)

def _rgb_to_gray(image: torch.Tensor) -> torch.Tensor:
    if image.shape[0] == 1:
        return image
    return (image * _RGB_TO_GRAY_WEIGHTS.to(image.device, dtype=image.dtype)).sum(dim=0, keepdim=True)

def _resize_tensor_image(
    image: torch.Tensor,
    resize: Optional[Union[int, Tuple[int, int]]],
    df: Optional[int],
    resize_by_stretch: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    h, w = image.shape[-2], image.shape[-1]
    if resize_by_stretch:
        if resize is None:
            w_new, h_new = w, h
        else:
            w_new, h_new = (resize, resize) if isinstance(resize, int) else (resize[1], resize[0])
    else:
        if resize:
            resize_val = resize
            if not isinstance(resize_val, int):
                assert resize_val[0] == resize_val[1]
                resize_val = resize_val[0]
            w_new, h_new = get_resized_wh(w, h, resize_val)
            w_new, h_new = get_divisible_wh(w_new, h_new, df)
        else:
            w_new, h_new = w, h

    if (h_new, w_new) != (h, w):
        image = F.interpolate(
            image.unsqueeze(0),
            size=(h_new, w_new),
            mode='bilinear',
            align_corners=False,
        ).squeeze(0)

    scale = torch.tensor([w / w_new, h / h_new], dtype=torch.float)
    origin_img_size = torch.tensor([h, w], dtype=torch.float)
    return image, scale, origin_img_size

def _scale_homography(
    homography: torch.Tensor,
    scale_src: torch.Tensor,
    scale_dst: torch.Tensor,
) -> torch.Tensor:
    sx0, sy0 = 1.0 / scale_src[0].item(), 1.0 / scale_src[1].item()
    sx1, sy1 = 1.0 / scale_dst[0].item(), 1.0 / scale_dst[1].item()
    s0 = torch.tensor([[sx0, 0.0, 0.0], [0.0, sy0, 0.0], [0.0, 0.0, 1.0]], dtype=homography.dtype)
    s1 = torch.tensor([[sx1, 0.0, 0.0], [0.0, sy1, 0.0], [0.0, 0.0, 1.0]], dtype=homography.dtype)
    return s1 @ homography @ torch.linalg.inv(s0)

class SatelliteSeasonalHomographyDataset(Dataset):
    def __init__(self,
                 maps_path: str,
                 num_samples: int = 1000,
                 crop_resolution: Tuple[int, int] = (256, 256),
                 map_to_img_ratio: int = 4,
                 min_size_meters: float = 50.0,
                 max_size_meters: float = 100.0,
                 px_per_meter: float = 4.0,
                 mode: str = 'train',
                 min_pixel_variance: Optional[float] = None,
                 max_variance_resamples: int = 10,
                 augmentation: Optional[AugmentationPipeline] = None,
                 use_augmentation: bool = False):

        self.train = (mode == 'train')
        self.num_samples = num_samples
        self.crop_res = crop_resolution  # Target size: (H, W)
        self.map_to_img_ratio = map_to_img_ratio
        self.map_res = (crop_resolution[0] * map_to_img_ratio,
                        crop_resolution[1] * map_to_img_ratio)
        
        # Sampling ranges
        self.angle_range = (-55.0, 55.0)
        self.scale_range = (0.65, 1.35)
        self.translation_range = (-0.30, 0.30)

        # zero translation
        self.zero_for_test = False

        list_of_images_paths = sorted(list(Path(maps_path).glob('*.tif')))

        if len(list_of_images_paths) < 2:
            raise ValueError(f"Need at least 2 seasonal images in {maps_path}.")

        self.list_of_images = [pyvips.Image.new_from_file(
            str(el)) for el in list_of_images_paths]

        self.org_width = self.list_of_images[0].width
        self.org_height = self.list_of_images[0].height

        self.min_size = int(min_size_meters * px_per_meter)
        self.max_size = int(max_size_meters * px_per_meter)
        self.rng = np.random.default_rng(seed=42)
        self.min_pixel_variance = min_pixel_variance
        self.max_variance_resamples = max_variance_resamples
        self.augmentation = augmentation
        self.use_augmentation = use_augmentation

    def sample_map_coords(self, map_size: Tuple[int, int]) -> Tuple[int, int]:
        """Samples top-left coordinates for cropping a map of given size."""
        max_x = self.org_width - map_size[1]
        max_y = self.org_height - map_size[0]
        
        if max_x < 0 or max_y < 0:
             raise ValueError(f"Map size {map_size} is larger than original image size ({self.org_width}, {self.org_height})")

        x = self.rng.integers(0, max_x + 1)
        y = self.rng.integers(0, max_y + 1)
        return x, y

    def get_homography(self, h: int, w: int) -> torch.Tensor:
        """Generates a 3x3 Homography following the Kornia signature."""

        if self.zero_for_test:
            return torch.eye(3).unsqueeze(0)

        # 1. Sample parameters
        angle_val = self.rng.uniform(*self.angle_range)
        scale_val = self.rng.uniform(*self.scale_range)
        tx = self.rng.uniform(*self.translation_range) * w
        ty = self.rng.uniform(*self.translation_range) * h

        # 2. Prepare Tensors according to the Docstring
        # center: (B, 2)
        center = torch.tensor([[w / 2.0, h / 2.0]], dtype=torch.float32)
        # angle: (B)
        angle = torch.tensor([angle_val], dtype=torch.float32)
        # scale: (B, 2) for x,y scaling
        scale = torch.tensor([[scale_val, scale_val]], dtype=torch.float32)

        # 3. Get Affine Matrix (1, 2, 3)
        M = kornia.geometry.transform.get_rotation_matrix2d(
            center, angle, scale)

        # 4. Apply Translation to the last column (tx, ty)
        M[0, 0, 2] += tx
        M[0, 1, 2] += ty

        # 5. Convert 2x3 Affine to 3x3 Homography
        H = kornia.geometry.conversions.convert_affinematrix_to_homography(M)
        return H

    def __len__(self):
        return self.num_samples

    def _has_sufficient_variance(self, ten_A: torch.Tensor, ten_B: torch.Tensor) -> bool:
        if self.min_pixel_variance is None:
            return True
        variance_a = torch.var(ten_A, unbiased=False).item()
        variance_b = torch.var(ten_B, unbiased=False).item()
        return variance_a >= self.min_pixel_variance and variance_b >= self.min_pixel_variance

    def _generate_sample(self) -> Tuple[Dict[str, Union[torch.Tensor, str]], torch.Tensor, torch.Tensor]:
        # 1. Pick two different seasons
        if self.train:
            idx1, idx2 = self.rng.choice(len(self.list_of_images), 2, replace=False)
        else:
            idx1 = self.rng.integers(0, len(self.list_of_images) - 1)
            # always last season for testing
            idx2 = len(self.list_of_images) - 1

        crop_size = self.rng.integers(
            self.min_size, self.max_size)
        map_sample_size = crop_size * self.map_to_img_ratio

        x_map, y_map = self.sample_map_coords(
            (map_sample_size, map_sample_size))

        # 3. Fetch from Tiff and convert to Tensors [C, H, W]
        raw_A = self.list_of_images[idx1].crop(
            x_map, y_map, map_sample_size, map_sample_size).numpy()
        raw_B = self.list_of_images[idx2].crop(
            x_map, y_map, map_sample_size, map_sample_size).numpy()

        ten_A = torch.from_numpy(raw_A).permute(2, 0, 1).float() / 255.0
        ten_B = torch.from_numpy(raw_B).permute(2, 0, 1).float() / 255.0

        # resize to map_crop_resolution
        ten_A = F.interpolate(ten_A.unsqueeze(0),
                              size=self.map_res, mode='bilinear',
                              align_corners=False,
                              ).squeeze(0)

        ten_B = F.interpolate(ten_B.unsqueeze(0),
                              size=self.map_res, mode='bilinear',
                              align_corners=False,
                              ).squeeze(0)

        # 4. Generate Transformation (calculated relative to the fetched area)
        H = self.get_homography(self.map_res[0], self.map_res[1])

        # 5. Warp Season B and mask into the frame of Season A
        # dsize=fetch_size keeps it at the same scale before center cropping
        ten_B_warped = kornia.geometry.transform.warp_perspective(
            ten_B.unsqueeze(0),
            H,
            dsize=self.map_res,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False,
        ).squeeze(0)

        # Calculate gt_warp and mask
        H_inv = torch.linalg.inv(H)
        H_A, W_A = self.crop_res
        H_B, W_B = self.map_res

        # Create grid for im_A (pixels)
        grid_A = kornia.utils.create_meshgrid(
            H_A, W_A, normalized_coordinates=False, device=ten_A.device)

        # Adjust for crop offset
        offset_x = (W_B - W_A) / 2
        offset_y = (H_B - H_A) / 2

        grid_A[..., 0] += offset_x
        grid_A[..., 1] += offset_y

        # Apply H_inv to get coordinates in im_B
        points_warped = grid_A.reshape(1, -1, 2)
        points_B = kornia.geometry.linalg.transform_points(
            H_inv.unsqueeze(0), points_warped)
        grid_B = points_B.reshape(1, H_A, W_A, 2)

        # Normalize to [-1, 1] for im_B
        gt_warp = grid_B.clone()
        gt_warp[..., 0] = 2 * (grid_B[..., 0] + 0.5) / W_B - 1
        gt_warp[..., 1] = 2 * (grid_B[..., 1] + 0.5) / H_B - 1
        gt_warp = gt_warp.squeeze(0)

        # Mask valid regions
        mask = (grid_B[..., 0] >= 0) & (grid_B[..., 0] < W_B) & \
               (grid_B[..., 1] >= 0) & (grid_B[..., 1] < H_B)
        mask = mask.float().squeeze(0)

        # map image B (large)
        im_B = ten_A
        # drone image A (small) crop after warp
        im_A = kornia.geometry.transform.center_crop(
            ten_B_warped.unsqueeze(0), self.crop_res).squeeze(0)

        sample = {
            "im_A": im_A,
            "im_A_label": f'season_{idx2}_x{x_map}_y{y_map}_size{map_sample_size}',
            "im_B": im_B,
            "im_B_label": f'season_{idx1}_x{x_map}_y{y_map}_size{map_sample_size}',
            "gt_warp": gt_warp,
            "mask": mask,
            "homography": H.squeeze(0)
        }

        return sample, ten_A, ten_B

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str]]:
        last_sample = None
        max_attempts = max(self.max_variance_resamples, 0)
        for attempt in range(max_attempts + 1):
            sample, ten_A, ten_B = self._generate_sample()
            if self.use_augmentation and self.augmentation is not None:
                sample = self.augmentation(sample)
            last_sample = sample
            if self._has_sufficient_variance(ten_A, ten_B):
                return sample

        return last_sample


class SatelliteSeasonalHomographyCommonDataset(SatelliteSeasonalHomographyDataset):
    def __init__(
        self,
        maps_path: str,
        num_samples: int = 1000,
        crop_resolution: Tuple[int, int] = (1024, 1024),
        map_to_img_ratio: int = 4,
        min_size_meters: float = 50.0,
        max_size_meters: float = 100.0,
        px_per_meter: float = 4.0,
        mode: str = 'test',
        img_resize: Optional[Union[int, Tuple[int, int]]] = None,
        df: Optional[int] = None,
        img_padding: bool = False,
        depth_padding: bool = True,
        fp16: bool = False,
        load_origin_rgb: bool = True,
        read_gray: bool = True,
        normalize_img: bool = False,
        resize_by_stretch: bool = False,
        gt_matches_padding_n: int = 100,
    ):
        super().__init__(
            maps_path=maps_path,
            num_samples=num_samples,
            crop_resolution=crop_resolution,
            map_to_img_ratio=map_to_img_ratio,
            min_size_meters=min_size_meters,
            max_size_meters=max_size_meters,
            px_per_meter=px_per_meter,
            mode=mode,
            min_pixel_variance=0.002,
            max_variance_resamples=10,
            augmentation=None,
            use_augmentation=False,
        )
        self.img_resize = img_resize
        self.df = df
        self.img_padding = img_padding
        self.depth_max_size = 4000 if depth_padding else 2000
        self.fp16 = fp16
        self.load_origin_rgb = load_origin_rgb
        self.read_gray = read_gray
        self.normalize_img = normalize_img
        self.resize_by_stretch = resize_by_stretch
        self.gt_matches_padding_n = gt_matches_padding_n
        self.dataset_name = "SAT"
        self.scene_id = Path(maps_path).name

    def _sample_pair(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str, str]:
        if self.train:
            idx1, idx2 = self.rng.choice(len(self.list_of_images), 2, replace=False)
        else:
            idx1 = self.rng.integers(0, len(self.list_of_images) - 1)
            idx2 = len(self.list_of_images) - 1

        crop_size = self.rng.integers(self.min_size, self.max_size)
        map_sample_size = crop_size * self.map_to_img_ratio
        x_map, y_map = self.sample_map_coords((map_sample_size, map_sample_size))

        raw_A = self.list_of_images[idx1].crop(
            x_map, y_map, map_sample_size, map_sample_size).numpy()
        raw_B = self.list_of_images[idx2].crop(
            x_map, y_map, map_sample_size, map_sample_size).numpy()

        ten_A = torch.from_numpy(raw_A).permute(2, 0, 1).float() / 255.0
        ten_B = torch.from_numpy(raw_B).permute(2, 0, 1).float() / 255.0

        ten_A = F.interpolate(
            ten_A.unsqueeze(0),
            size=self.map_res,
            mode='bilinear',
            align_corners=False,
        ).squeeze(0)
        ten_B = F.interpolate(
            ten_B.unsqueeze(0),
            size=self.map_res,
            mode='bilinear',
            align_corners=False,
        ).squeeze(0)

        homography = self.get_homography(self.map_res[0], self.map_res[1]).squeeze(0)

        label_a = f"sat/season_{idx1}_x{x_map}_y{y_map}_size{map_sample_size}"
        label_b = f"sat/season_{idx2}_x{x_map}_y{y_map}_size{map_sample_size}"
        return ten_B, ten_A, homography, label_b, label_a

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str]]:
        image0_rgb, image1_rgb, homography, label0, label1 = self._sample_pair()

        origin_img_size0 = torch.tensor([image0_rgb.shape[1], image0_rgb.shape[2]], dtype=torch.float)
        origin_img_size1 = torch.tensor([image1_rgb.shape[1], image1_rgb.shape[2]], dtype=torch.float)

        image0_resized, scale0, _ = _resize_tensor_image(
            image0_rgb, self.img_resize, self.df, self.resize_by_stretch)
        image1_resized, scale1, _ = _resize_tensor_image(
            image1_rgb, self.img_resize, self.df, self.resize_by_stretch)

        if self.img_padding:
            pad_to = max(image0_resized.shape[-2], image0_resized.shape[-1],
                         image1_resized.shape[-2], image1_resized.shape[-1])
            image0_np, mask0 = pad_bottom_right(image0_resized.numpy(), pad_to, ret_mask=True)
            image1_np, mask1 = pad_bottom_right(image1_resized.numpy(), pad_to, ret_mask=True)
            image0_resized = torch.from_numpy(image0_np)
            image1_resized = torch.from_numpy(image1_np)
            mask0 = torch.from_numpy(mask0)
            mask1 = torch.from_numpy(mask1)
        else:
            mask0 = None
            mask1 = None

        if self.read_gray:
            image0 = _rgb_to_gray(image0_resized)
            image1 = _rgb_to_gray(image1_resized)
        else:
            image0 = image0_resized
            image1 = image1_resized

        if not self.read_gray and self.normalize_img:
            normalizer = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            image0 = normalizer(image0)
            image1 = normalizer(image1)

        homography_resized = _scale_homography(homography, scale0, scale1)

        depth0 = torch.zeros([self.depth_max_size, self.depth_max_size], dtype=torch.float)
        depth1 = torch.zeros([self.depth_max_size, self.depth_max_size], dtype=torch.float)
        homo_mask0 = torch.zeros((1, image0.shape[-2], image0.shape[-1]))
        homo_mask1 = torch.zeros((1, image1.shape[-2], image1.shape[-1]))
        gt_matches = torch.zeros((self.gt_matches_padding_n, 4), dtype=torch.float)

        T_0to1 = T_1to0 = torch.zeros((4, 4), dtype=torch.float)
        K_0 = torch.zeros((3, 3), dtype=torch.float)
        K_1 = torch.zeros((3, 3), dtype=torch.float)

        data = {
            'image0': image0.half() if self.fp16 else image0,
            'depth0': depth0.half() if self.fp16 else depth0,
            'image1': image1.half() if self.fp16 else image1,
            'depth1': depth1.half() if self.fp16 else depth1,
            'T_0to1': T_0to1,
            'T_1to0': T_1to0,
            'K0': K_0,
            'K1': K_1,
            'homo_mask0': homo_mask0,
            'homo_mask1': homo_mask1,
            'homography': homography_resized.to(torch.float32),
            'norm_pixel_mat': torch.zeros((3, 3), dtype=torch.float),
            'homo_sample_normed': torch.zeros((3, 3), dtype=torch.float),
            'gt_matches': gt_matches,
            'gt_matches_mask': torch.zeros((1,), dtype=torch.bool),
            'origin_img_size0': origin_img_size0,
            'origin_img_size1': origin_img_size1,
            'scale0': scale0.half() if self.fp16 else scale0,
            'scale1': scale1.half() if self.fp16 else scale1,
            'dataset_name': self.dataset_name,
            'scene_id': self.scene_id,
            'pair_id': idx,
            'pair_names': (label0, label1),
            'rel_pair_names': (label0, label1),
        }

        if mask0 is not None:
            data.update({'mask0': mask0, 'mask1': mask1})

        if self.load_origin_rgb:
            data.update({
                'image0_rgb_origin': image0_rgb,
                'image1_rgb_origin': image1_rgb,
            })

        return data


if __name__ == "__main__":
    path_config = Path(__file__).parent.parent.parent / 'experiments' / 'aug_config' / 'config.yaml'
    augmentation = AugmentationPipeline(AugmentationConfig.from_yaml(path_config))
    dataset = SatelliteSeasonalHomographyDataset(
        maps_path='../vps_n/sat_data/',
        num_samples=1000,
        augmentation=augmentation,
        use_augmentation=True,
    )

    # # benchmark dataset loading
    
    # for worker in [0]:
    #     import time
    #     start_time = time.time()
    #     dataloader = torch.utils.data.DataLoader(
    #         dataset,
    #         batch_size=8,
    #         num_workers=worker,
    #     )
    #     for i, sample in enumerate(dataloader):
    #         pass
    #     end_time = time.time()
    #     print(f"DataLoader with {worker} workers took {end_time - start_time:.2f} seconds.")
    #     print(f"images per second: {len(dataset)/(end_time - start_time):.2f}")
    
    
    import os

    for i in range(len(dataset)):
        sample = dataset[i]
        # save to path
        
        save_path = f"../vps_n/sat_data/samples/sample_{i}.pt"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(sample, save_path)
        # print shape
        print(
            f"Sample {i}: im_A shape: {sample['im_A'].shape}, im_B shape: {sample['im_B'].shape}")

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3)
        # fig size 12x8
        fig.set_size_inches(12, 8)
        ax[0].imshow(sample["im_A"].permute(1, 2, 0))
        
        ax[0].set_title(sample["im_A_label"])

        ax[1].imshow(sample["im_B"].permute(1, 2, 0))
        ax[1].set_title(sample["im_B_label"])

        img_A_upscale = torch.zeros_like(sample["im_B"])
        c_h, c_w = sample["im_A"].shape[1], sample["im_A"].shape[2]
        
        start_h = (img_A_upscale.shape[1] - c_h) // 2
        start_w = (img_A_upscale.shape[2] - c_w) // 2
        img_A_upscale[:, start_h:start_h + c_h,
                      start_w:start_w + c_w] = sample["im_A"]
        # add RED border to img_A_upscale
        img_A_upscale[0, start_h:start_h + c_h, start_w] = 1.0
        img_A_upscale[0, start_h:start_h + c_h, start_w + c_w - 1] = 1.0
        img_A_upscale[0, start_h, start_w:start_w + c_w] = 1.0
        img_A_upscale[0, start_h + c_h - 1, start_w:start_w + c_w] = 1.0
        H_inv = torch.linalg.inv(sample["homography"]).unsqueeze(0)
        warped_A = kornia.geometry.transform.warp_perspective(
            img_A_upscale.unsqueeze(0),
            H_inv,
            dsize=(img_A_upscale.shape[1], img_A_upscale.shape[2]),
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False,
        ).squeeze(0)

        mask = (warped_A.sum(
            dim=0) != 0).float()  # simple mask where any channel is non-zero

        combined = sample["im_B"] * (1 - mask) + warped_A * mask

        ax[2].imshow(combined.permute(1, 2, 0))

        plt.show()
