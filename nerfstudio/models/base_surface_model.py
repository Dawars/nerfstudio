# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Implementation of Base surface model.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Tuple, Type, cast

import torch
import torch.nn.functional as F
from torch.nn import Parameter

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.field_components.encodings import NeRFEncoding
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.fields.nerfacto_field import NerfactoField
from nerfstudio.fields.sdf_field import SDFFieldConfig
from nerfstudio.fields.vanilla_nerf_field import NeRFField
from nerfstudio.model_components.losses import L1Loss, MSELoss, ScaleAndShiftInvariantLoss, monosdf_normal_loss
from nerfstudio.model_components.ray_samplers import LinearDisparitySampler
from nerfstudio.model_components.renderers import AccumulationRenderer, DepthRenderer, RGBRenderer, SemanticRenderer
from nerfstudio.model_components.scene_colliders import AABBBoxCollider, NearFarCollider, SphereCollider
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import colormaps
from nerfstudio.utils.colors import get_color
from nerfstudio.utils.math import normalized_depth_scale_and_shift


@dataclass
class SurfaceModelConfig(ModelConfig):
    """Surface Model Config"""

    _target: Type = field(default_factory=lambda: SurfaceModel)
    near_plane: float = 0.05
    """How far along the ray to start sampling."""
    far_plane: float = 4.0
    """How far along the ray to stop sampling."""
    far_plane_bg: float = 1000.0
    """How far along the ray to stop sampling of the background model."""
    background_color: Literal["random", "last_sample", "white", "black"] = "black"
    """Whether to randomize the background color."""
    use_average_appearance_embedding: bool = False
    """Whether to use average appearance embedding or zeros for inference."""
    eikonal_loss_mult: float = 0.1
    """Monocular normal consistency loss multiplier."""
    fg_mask_loss_mult: float = 0.01
    """Foreground mask loss multiplier."""
    mono_normal_loss_mult: float = 0.0
    """Monocular normal consistency loss multiplier."""
    mono_depth_loss_mult: float = 0.0
    """Monocular depth consistency loss multiplier."""
    patch_warp_loss_mult: float = 0.0
    """Multi-view consistency warping loss multiplier."""
    patch_size: int = 11
    """Multi-view consistency warping loss patch size."""
    patch_warp_angle_thres: float = 0.3
    """Threshold for valid homograph of multi-view consistency warping loss"""
    min_patch_variance: float = 0.01
    """Threshold for minimal patch variance"""
    topk: int = 4
    """Number of minimal patch consistency selected for training"""
    sensor_depth_truncation: float = 0.015
    """Sensor depth trunction, default value is 0.015 which means 5cm with a rough scale value 0.3 (0.015 = 0.05 * 0.3)"""
    sensor_depth_l1_loss_mult: float = 0.0
    """Sensor depth L1 loss multiplier."""
    sensor_depth_freespace_loss_mult: float = 0.0
    """Sensor depth free space loss multiplier."""
    sensor_depth_sdf_loss_mult: float = 0.0
    """Sensor depth sdf loss multiplier."""
    sparse_points_sdf_loss_mult: float = 0.0
    """sparse point sdf loss multiplier"""
    s3im_loss_mult: float = 0.0
    """S3IM loss multiplier."""
    s3im_kernel_size: int = 4
    """S3IM kernel size."""
    s3im_stride: int = 4
    """S3IM stride."""
    s3im_repeat_time: int = 10
    """S3IM repeat time."""
    s3im_patch_height: int = 32
    """S3IM virtual patch height."""
    sdf_field: SDFFieldConfig = field(default_factory=SDFFieldConfig)
    """Config for SDF Field"""
    background_model: Literal["grid", "mlp", "none"] = "mlp"
    """background models"""
    num_samples_outside: int = 32
    """Number of samples outside the bounding sphere for background"""
    periodic_tvl_mult: float = 0.0
    """Total variational loss multiplier"""
    overwrite_near_far_plane: bool = False
    """whether to use near and far collider from command line"""
    scene_contraction_norm: Literal["inf", "l2"] = "inf"
    """Which norm to use for the scene contraction."""
    color_loss: bool = False
    """Projecting mlp output to grayscale in rgb loss when input is grayscale"""


class SurfaceModel(Model):
    """Base surface model

    Args:
        config: Base surface model configuration to instantiate model
    """

    config: SurfaceModelConfig

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()

        if self.config.scene_contraction_norm == "inf":
            order = float("inf")
        elif self.config.scene_contraction_norm == "l2":
            order = None
        else:
            raise ValueError("Invalid scene contraction norm")

        self.scene_contraction = SceneContraction(order=order)
        # Can we also use contraction for sdf?
        # Fields
        self.field = self.config.sdf_field.setup(
            aabb=self.scene_box.aabb,
            spatial_distortion=self.scene_contraction,
            num_images=self.num_train_data,
            use_average_appearance_embedding=self.config.use_average_appearance_embedding,
        )

        # Collider
        # if self.scene_box.collider_type == "near_far":
        #     self.collider = NearFarCollider(near_plane=self.scene_box.near, far_plane=self.scene_box.far)
        # elif self.scene_box.collider_type == "box":
        #     self.collider = AABBBoxCollider(self.scene_box, near_plane=self.scene_box.near)
        # elif self.scene_box.collider_type == "sphere":
        #     # TODO do we also use near if the ray don't intersect with the sphere
        #     self.collider = SphereCollider(radius=self.scene_box.radius, soft_intersection=True)
        # else:
        #     raise NotImplementedError
        # Neural Reconstruction in the wild use sphere collider so we overwrite it here
        self.collider = SphereCollider(center=torch.zeros([3,]), radius=1.0, soft_intersection=False)

        # command line near and far has highest priority
        if self.config.overwrite_near_far_plane:
            self.collider = NearFarCollider(near_plane=self.config.near_plane, far_plane=self.config.far_plane)

        # background model
        if self.config.background_model == "grid":
            self.field_background = NerfactoField(
                self.scene_box.aabb,
                spatial_distortion=self.scene_contraction,
                num_images=self.num_train_data,
                use_average_appearance_embedding=self.config.use_average_appearance_embedding,
            )
        elif self.config.background_model == "mlp":
            position_encoding = NeRFEncoding(
                in_dim=3, num_frequencies=10, min_freq_exp=0.0, max_freq_exp=9.0, include_input=True
            )
            direction_encoding = NeRFEncoding(
                in_dim=3, num_frequencies=4, min_freq_exp=0.0, max_freq_exp=3.0, include_input=True
            )

            self.field_background = NeRFField(
                position_encoding=position_encoding,
                direction_encoding=direction_encoding,
                spatial_distortion=self.scene_contraction,
            )
        else:
            # dummy background model
            self.field_background = Parameter(torch.ones(1), requires_grad=False)

        self.sampler_bg = LinearDisparitySampler(num_samples=self.config.num_samples_outside)

        # renderers
        background_color = (
            get_color(self.config.background_color)
            if self.config.background_color in set(["white", "black"])
            else self.config.background_color
        )
        self.renderer_rgb = RGBRenderer(background_color=background_color)
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer(method="expected")
        self.renderer_normal = SemanticRenderer()
        # patch warping
        # self.patch_warping = PatchWarping(
        #     patch_size=self.config.patch_size, valid_angle_thres=self.config.patch_warp_angle_thres
        # )
        # losses
        self.rgb_loss = L1Loss()
        # self.s3im_loss = S3IM(s3im_kernel_size=self.config.s3im_kernel_size, s3im_stride=self.config.s3im_stride, s3im_repeat_time=self.config.s3im_repeat_time, s3im_patch_height=self.config.s3im_patch_height)

        self.eikonal_loss = MSELoss()
        self.depth_loss = ScaleAndShiftInvariantLoss(alpha=0.5, scales=1)
        self.sky_loss = torch.nn.BCEWithLogitsLoss()
        # self.patch_loss = MultiViewLoss(
        #     patch_size=self.config.patch_size, topk=self.config.topk, min_patch_variance=self.config.min_patch_variance
        # )
        # self.sensor_depth_loss = SensorDepthLoss(truncation=self.config.sensor_depth_truncation)

        # metrics
        from torchmetrics.functional import structural_similarity_index_measure
        from torchmetrics.image import PeakSignalNoiseRatio
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity()

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        param_groups["fields"] = list(self.field.parameters())
        param_groups["field_background"] = (
            [self.field_background]
            if isinstance(self.field_background, Parameter)
            else list(self.field_background.parameters())
        )
        return param_groups

    @abstractmethod
    def sample_and_forward_field(self, ray_bundle: RayBundle) -> Dict[str, Any]:
        """Takes in a Ray Bundle and returns a dictionary of samples and field output.

        Args:
            ray_bundle: Input bundle of rays. This raybundle should have all the
            needed information to compute the outputs.

        Returns:
            Outputs of model. (ie. rendered colors)
        """

    def get_outputs(self, ray_bundle: RayBundle) -> Dict[str, torch.Tensor]:
        """Takes in a Ray Bundle and returns a dictionary of outputs.

        Args:
            ray_bundle: Input bundle of rays. This raybundle should have all the
            needed information to compute the outputs.

        Returns:
            Outputs of model. (ie. rendered colors)
        """
        assert ray_bundle.metadata is not None and "directions_norm" in ray_bundle.metadata, (
            "directions_norm is required in ray_bundle.metadata"
        )

        samples_and_field_outputs = self.sample_and_forward_field(ray_bundle=ray_bundle)

        # shortcuts
        field_outputs: Dict[FieldHeadNames, torch.Tensor] = cast(
            Dict[FieldHeadNames, torch.Tensor], samples_and_field_outputs["field_outputs"]
        )
        ray_samples = samples_and_field_outputs["ray_samples"]
        weights = samples_and_field_outputs["weights"]
        bg_transmittance = samples_and_field_outputs["bg_transmittance"]

        rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
        depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
        # the rendered depth is point-to-point distance and we should convert to depth
        depth = depth / ray_bundle.metadata["directions_norm"]

        normal = self.renderer_normal(semantics=field_outputs[FieldHeadNames.NORMAL], weights=weights)
        accumulation = self.renderer_accumulation(weights=weights)

        # background model
        if self.config.background_model != "none":
            assert isinstance(self.field_background, torch.nn.Module), "field_background should be a module"
            assert ray_bundle.fars is not None, "fars is required in ray_bundle"
            # sample inversely from far to 1000 and points and forward the bg model
            ray_bundle.nears = ray_bundle.fars
            assert ray_bundle.fars is not None
            ray_bundle.fars = torch.ones_like(ray_bundle.fars) * self.config.far_plane_bg

            ray_samples_bg = self.sampler_bg(ray_bundle)
            # use the same background model for both density field and occupancy field
            assert not isinstance(self.field_background, Parameter)
            field_outputs_bg = self.field_background(ray_samples_bg)
            weights_bg = ray_samples_bg.get_weights(field_outputs_bg[FieldHeadNames.DENSITY])

            rgb_bg = self.renderer_rgb(rgb=field_outputs_bg[FieldHeadNames.RGB], weights=weights_bg)
            depth_bg = self.renderer_depth(weights=weights_bg, ray_samples=ray_samples_bg)
            accumulation_bg = self.renderer_accumulation(weights=weights_bg)

            # merge background color to foregound color
            rgb = rgb + bg_transmittance * rgb_bg

            bg_outputs = {
                "bg_rgb": rgb_bg,
                "bg_accumulation": accumulation_bg,
                "bg_depth": depth_bg,
                "bg_weights": weights_bg,
            }
        else:
            bg_outputs = {}

        outputs = {
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
            "normal": normal,
            "weights": weights,
            # used to scale z_vals for free space and sdf loss
            "directions_norm": ray_bundle.metadata["directions_norm"],
        }
        outputs.update(bg_outputs)

        if self.training:
            grad_points = field_outputs[FieldHeadNames.GRADIENT]
            outputs.update({"eik_grad": grad_points})
            outputs.update(samples_and_field_outputs)

        if "weights_list" in samples_and_field_outputs:
            weights_list = cast(List[torch.Tensor], samples_and_field_outputs["weights_list"])
            ray_samples_list = cast(List[torch.Tensor], samples_and_field_outputs["ray_samples_list"])

            for i in range(len(weights_list) - 1):
                outputs[f"prop_depth_{i}"] = self.renderer_depth(
                    weights=weights_list[i], ray_samples=ray_samples_list[i]
                )
        # this is used only in viewer
        outputs["normal_vis"] = (outputs["normal"] + 1.0) / 2.0
        return outputs

    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        """Computes and returns the losses dict.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
            metrics_dict: dictionary of metrics, some of which we can use for loss
        """
        loss_dict = {}
        image = batch["image"].to(self.device)
        pred_image, image = self.renderer_rgb.blend_background_for_loss_computation(
            pred_image=outputs["rgb"],
            pred_accumulation=outputs["accumulation"],
            gt_image=image,
        )

        # color loss
        if self.config.color_loss:
            # convert output to grayscale if input image is grayscale
            grayscale = batch["is_gray"][:, 0]
            if grayscale.any():  # https://openaccess.thecvf.com/content_cvpr_2017/papers/Nguyen_Why_You_Should_CVPR_2017_paper.pdf
                rgb2gray = image[grayscale][:, 0] * 0.2126 + \
                           image[grayscale][:, 1] * 0.7152 + \
                           image[grayscale][:, 2] * 0.0722

                image[grayscale] = rgb2gray.unsqueeze(-1)

        loss_dict["rgb_loss"] = self.rgb_loss(image, pred_image)
        if self.training:
            # eikonal loss
            grad_theta = outputs["eik_grad"]
            loss_dict["eikonal_loss"] = ((grad_theta.norm(2, dim=-1) - 1) ** 2).mean() * self.config.eikonal_loss_mult

            # sky loss
            if "semantics" in batch and self.config.fg_mask_loss_mult > 0:
                # sky loss
                fg_label = (batch["semantics"] != 2).float().to(self.device)  # sky
                weights_sum = outputs["weights"].sum(dim=1).clip(1e-3, 1.0 - 1e-3)
                loss_dict[f"fg_mask_loss"] = (
                        self.sky_loss(weights_sum, fg_label) * self.config.fg_mask_loss_mult
                )
            # sparse points sdf loss
            if "sparse_sfm_points" in batch and self.config.sparse_points_sdf_loss_mult > 0.0:
                sparse_sfm_points = batch["sparse_sfm_points"].to(self.device)  # Nx3
                sparse_sfm_points_sdf = self.field.forward_geonetwork(sparse_sfm_points[:, :3])[:, 0].contiguous()
                loss_dict["sparse_sfm_points_sdf_loss"] = (
                        torch.mean(torch.abs(sparse_sfm_points_sdf)) * self.config.sparse_points_sdf_loss_mult
                )

            # monocular normal loss
            if "normal" in batch and self.config.mono_normal_loss_mult > 0.0:
                normal_gt = batch["normal"].to(self.device)
                normal_pred = outputs["normal"]
                loss_dict["normal_loss"] = (
                    monosdf_normal_loss(normal_pred, normal_gt) * self.config.mono_normal_loss_mult
                )

            # monocular depth loss
            if "depth" in batch and self.config.mono_depth_loss_mult > 0.0:
                depth_gt = batch["depth"].to(self.device)[..., None]
                depth_pred = outputs["depth"]

                mask = torch.ones_like(depth_gt).reshape(1, 32, -1).bool()
                loss_dict["depth_loss"] = (
                    self.depth_loss(depth_pred.reshape(1, 32, -1), (depth_gt * 50 + 0.5).reshape(1, 32, -1), mask)
                    * self.config.mono_depth_loss_mult
                )

        return loss_dict

    def get_metrics_dict(self, outputs, batch) -> Dict[str, torch.Tensor]:
        """Compute and returns metrics.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
        """
        metrics_dict = {}
        image = batch["image"].to(self.device)
        image = self.renderer_rgb.blend_background(image)
        metrics_dict["psnr"] = self.psnr(outputs["rgb"], image)
        return metrics_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        """Writes the test image outputs.
        Args:
            outputs: Outputs of the model.
            batch: Batch of data.

        Returns:
            A dictionary of metrics.
        """
        image = batch["image"].to(self.device)
        image = self.renderer_rgb.blend_background(image)
        rgb = outputs["rgb"]
        acc = colormaps.apply_colormap(outputs["accumulation"])

        normal = outputs["normal"]
        normal = (normal + 1.0) / 2.0

        combined_rgb = torch.cat([image, rgb], dim=1)
        combined_acc = torch.cat([acc], dim=1)
        if "depth" in batch:
            depth_gt = batch["depth"].to(self.device)
            depth_pred = outputs["depth"]

            # align to predicted depth and normalize
            scale, shift = normalized_depth_scale_and_shift(
                depth_pred[None, ..., 0], depth_gt[None, ...], depth_gt[None, ...] > 0.0
            )
            depth_pred = depth_pred * scale + shift

            combined_depth = torch.cat([depth_gt[..., None], depth_pred], dim=1)
            combined_depth = colormaps.apply_depth_colormap(combined_depth)
        else:
            depth = colormaps.apply_depth_colormap(
                outputs["depth"],
                accumulation=outputs["accumulation"],
            )
            combined_depth = torch.cat([depth], dim=1)

        if "normal" in batch:
            normal_gt = (batch["normal"].to(self.device) + 1.0) / 2.0
            combined_normal = torch.cat([normal_gt, normal], dim=1)
        else:
            combined_normal = torch.cat([normal], dim=1)

        images_dict = {
            "img": combined_rgb,
            "accumulation": combined_acc,
            "depth": combined_depth,
            "normal": combined_normal,
        }

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        image = torch.moveaxis(image, -1, 0)[None, ...]
        rgb = torch.moveaxis(rgb, -1, 0)[None, ...]

        psnr = self.psnr(image, rgb)
        ssim = self.ssim(image, rgb)
        lpips = self.lpips(image, rgb)

        # all of these metrics will be logged as scalars
        metrics_dict = {"psnr": float(psnr.item()), "ssim": float(ssim)}  # type: ignore
        metrics_dict["lpips"] = float(lpips)

        return metrics_dict, images_dict
