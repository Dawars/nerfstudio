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
Field for SDF based model, rather then estimating density to generate a surface,
a signed distance function (SDF) for surface representation is used to help with extracting high fidelity surfaces
"""
import math
from dataclasses import dataclass, field
from typing import Dict, Literal, Optional, Type

import numpy as np
import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor, nn
from torch.nn.parameter import Parameter

from nerfstudio.cameras.rays import RaySamples
from nerfstudio.field_components.embedding import Embedding
from nerfstudio.field_components.encodings import NeRFEncoding
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SpatialDistortion
from nerfstudio.fields.base_field import Field, FieldConfig
from nerfstudio.utils.external import tcnn


class LearnedVariance(nn.Module):
    """Variance network in NeuS

    Args:
        init_val: initial value in NeuS variance network
    """

    variance: Tensor

    def __init__(self, init_val):
        super().__init__()
        self.register_parameter("variance", nn.Parameter(init_val * torch.ones(1), requires_grad=True))

    def forward(self, x: Float[Tensor, "1"]) -> Float[Tensor, "1"]:
        """Returns current variance value"""
        return torch.ones([len(x), 1], device=x.device) * torch.exp(self.variance * 10.0)

    def get_variance(self) -> Float[Tensor, "1"]:
        """return current variance value"""
        return torch.exp(self.variance * 10.0).clip(1e-6, 1e6)


@dataclass
class SDFFieldConfig(FieldConfig):
    """SDF Field Config"""

    _target: Type = field(default_factory=lambda: SDFField)
    num_layers: int = 8
    """Number of layers for geometric network"""
    hidden_dim: int = 256
    """Number of hidden dimension of geometric network"""
    geo_feat_dim: int = 256
    """Dimension of geometric feature"""
    num_layers_color: int = 4
    """Number of layers for color network"""
    hidden_dim_color: int = 256
    """Number of hidden dimension of color network"""
    appearance_embedding_dim: int = 32
    """Dimension of appearance embedding"""
    use_appearance_embedding: bool = False
    """Whether to use appearance embedding"""
    bias: float = 0.8
    """Sphere size of geometric initialization"""
    geometric_init: bool = True
    """Whether to use geometric initialization"""
    inside_outside: bool = True
    """Whether to revert signed distance value, set to True for indoor scene"""
    weight_norm: bool = True
    """Whether to use weight norm for linear layer"""
    use_grid_feature: bool = False
    """Whether to use multi-resolution feature grids"""
    divide_factor: float = 2.0
    """Normalization factor for multi-resolution grids"""
    beta_init: float = 0.1
    """Init learnable beta value for transformation of sdf to density"""
    encoding_type: Literal["hash", "periodic", "tensorf_vm"] = "hash"
    """feature grid encoding type"""
    position_encoding_max_degree: int = 6
    """positional encoding max degree"""
    use_diffuse_color: bool = False
    """whether to use diffuse color as in ref-nerf"""
    use_specular_tint: bool = False
    """whether to use specular tint as in ref-nerf"""
    use_reflections: bool = False
    """whether to use reflections as in ref-nerf"""
    use_n_dot_v: bool = False
    """whether to use n dot v as in ref-nerf"""
    rgb_padding: float = 0.001
    """Padding added to the RGB outputs"""
    off_axis: bool = False
    """whether to use off axis encoding from mipnerf360"""
    use_numerical_gradients: bool = False
    """whether to use numercial gradients"""
    num_levels: int = 16
    """number of levels for multi-resolution hash grids"""
    max_res: int = 2048
    """max resolution for multi-resolution hash grids"""
    base_res: int = 16
    """base resolution for multi-resolution hash grids"""
    log2_hashmap_size: int = 19
    """log2 hash map size for multi-resolution hash grids"""
    hash_features_per_level: int = 2
    """number of features per level for multi-resolution hash grids"""
    hash_smoothstep: bool = True
    """whether to use smoothstep for multi-resolution hash grids"""
    use_position_encoding: bool = True
    """whether to use positional encoding as input for geometric network"""

class LaplaceDensity(nn.Module):  # alpha * Laplace(loc=0, scale=beta).cdf(-sdf)
    """Laplace density from VolSDF"""

    def __init__(self, init_val, beta_min=0.0001):
        super().__init__()
        self.register_parameter("beta_min", nn.Parameter(beta_min * torch.ones(1), requires_grad=False))
        self.register_parameter("beta", nn.Parameter(init_val * torch.ones(1), requires_grad=True))

    def forward(
        self, sdf: Float[Tensor, "batch ... 1"], beta: Optional[Float[Tensor, "batch ... 1"]] = None
    ) -> Float[Tensor, "batch ... 1"]:
        """convert sdf value to density value with beta, if beta is missing, then use learnable beta"""

        if beta is None:
            beta = self.get_beta()

        alpha = 1.0 / beta
        return alpha * (0.5 + 0.5 * sdf.sign() * torch.expm1(-sdf.abs() / beta))

    def get_beta(self):
        """return current beta value"""
        beta = self.beta.abs() + self.beta_min
        return beta


class SigmoidDensity(nn.Module):  # alpha * Laplace(loc=0, scale=beta).cdf(-sdf)
    """Sigmoid density from VolSDF"""

    def __init__(self, init_val, beta_min=0.0001):
        super().__init__()
        self.register_parameter("beta_min", nn.Parameter(beta_min * torch.ones(1), requires_grad=False))
        self.register_parameter("beta", nn.Parameter(init_val * torch.ones(1), requires_grad=True))

    def forward(
        self, sdf: Float[Tensor, "batch ... 1"], beta: Optional[Float[Tensor, "batch ... 1"]] = None
    ) -> Float[Tensor, "batch ... 1"]:
        """convert sdf value to density value with beta, if beta is missing, then use learable beta"""

        if beta is None:
            beta = self.get_beta()

        alpha = 1.0 / beta

        # negtive sdf will have large density
        return alpha * torch.sigmoid(-sdf * alpha)

    def get_beta(self):
        """return current beta value"""
        beta = self.beta.abs() + self.beta_min
        return beta

class SingleVarianceNetwork(nn.Module):
    """Variance network in NeuS

    Args:
        nn (_type_): init value in NeuS variance network
    """

    def __init__(self, init_val):
        super(SingleVarianceNetwork, self).__init__()
        self.register_parameter("variance", nn.Parameter(init_val * torch.ones(1), requires_grad=True))

    def forward(self, x):
        """Returns current variance value"""
        return torch.ones([len(x), 1], device=x.device) * torch.exp(self.variance * 10.0)

    def get_variance(self):
        """return current variance value"""
        return torch.exp(self.variance * 10.0).clip(1e-6, 1e6)


class SDFField(Field):
    """
    A field for Signed Distance Functions (SDF).

    Args:
        config: The configuration for the SDF field.
        aabb: An axis-aligned bounding box for the SDF field.
        num_images: The number of images for embedding appearance.
        use_average_appearance_embedding: Whether to use average appearance embedding. Defaults to False.
        spatial_distortion: The spatial distortion. Defaults to None.
    """

    config: SDFFieldConfig

    def __init__(
        self,
        config: SDFFieldConfig,
        aabb: Float[Tensor, "2 3"],
        num_images: int,
        use_average_appearance_embedding: bool = False,
        spatial_distortion: Optional[SpatialDistortion] = None,
    ) -> None:
        super().__init__()
        self.config = config

        self.aabb = Parameter(aabb, requires_grad=False)

        self.spatial_distortion = spatial_distortion
        self.num_images = num_images

        self.embedding_appearance = Embedding(self.num_images, self.config.appearance_embedding_dim)
        self.use_average_appearance_embedding = use_average_appearance_embedding
        self.use_grid_feature = self.config.use_grid_feature
        self.divide_factor = self.config.divide_factor

        self.num_levels = self.config.num_levels
        self.max_res = self.config.max_res
        self.base_res = self.config.base_res
        self.log2_hashmap_size = self.config.log2_hashmap_size
        self.features_per_level = self.config.hash_features_per_level
        use_hash = True
        smoothstep = self.config.hash_smoothstep
        self.growth_factor = np.exp((np.log(config.max_res) - np.log(config.base_res)) / (config.num_levels - 1))

        if self.config.encoding_type == "hash":
            # feature encoding
            self.encoding = tcnn.Encoding(
                n_input_dims=3,
                encoding_config={
                    "otype": "HashGrid" if use_hash else "DenseGrid",
                    "n_levels": self.num_levels,
                    "n_features_per_level": self.features_per_level,
                    "log2_hashmap_size": self.log2_hashmap_size,
                    "base_resolution": self.base_res,
                    "per_level_scale": self.growth_factor,
                    "interpolation": "Smoothstep" if smoothstep else "Linear",
                },
            )
            self.hash_encoding_mask = torch.ones(
                self.num_levels * self.features_per_level,
                dtype=torch.float32,
            )

        else:
            raise NotImplementedError

        # we concat inputs position ourselves
        self.position_encoding = NeRFEncoding(
            in_dim=3,
            num_frequencies=self.config.position_encoding_max_degree,
            min_freq_exp=0.0,
            max_freq_exp=self.config.position_encoding_max_degree - 1,
            include_input=False,
            # off_axis=self.config.off_axis,  # mipnerf360
        )

        self.direction_encoding = NeRFEncoding(
            in_dim=3, num_frequencies=4, min_freq_exp=0.0, max_freq_exp=3.0, include_input=True
        )

        # initialize geometric network
        self.initialize_geo_layers()

        # laplace function for transform sdf to density from VolSDF
        self.laplace_density = LaplaceDensity(init_val=self.config.beta_init)
        # self.laplace_density = SigmoidDensity(init_val=self.config.beta_init)

        # deviation_network to compute alpha from sdf from NeuS
        self.deviation_network = LearnedVariance(init_val=self.config.beta_init)

        # diffuse and specular tint layer
        if self.config.use_diffuse_color:
            self.diffuse_color_pred = nn.Linear(self.config.geo_feat_dim, 3)
        if self.config.use_specular_tint:
            self.specular_tint_pred = nn.Linear(self.config.geo_feat_dim, 3)

        # color network
        dims = [self.config.hidden_dim_color for _ in range(self.config.num_layers_color)]
        if self.config.use_diffuse_color:
            in_dim = (
                self.direction_encoding.get_out_dim()
                + self.config.geo_feat_dim
                + self.embedding_appearance.get_out_dim()
            )
        else:
            # point, view_direction, normal, feature, embedding
            in_dim = (
                3
                + self.direction_encoding.get_out_dim()
                + 3
                + self.config.geo_feat_dim
                + self.embedding_appearance.get_out_dim()
            )
        if self.config.use_n_dot_v:
            in_dim += 1

        dims = [in_dim] + dims + [3]
        self.num_layers_color = len(dims)

        for layer in range(0, self.num_layers_color - 1):
            out_dim = dims[layer + 1]
            lin = nn.Linear(dims[layer], out_dim)
            torch.nn.init.kaiming_uniform_(lin.weight.data)
            torch.nn.init.zeros_(lin.bias.data)

            if self.config.weight_norm:
                lin = nn.utils.weight_norm(lin)
            setattr(self, "clin" + str(layer), lin)

        self.softplus = nn.Softplus(beta=100)
        self.relu = nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

        self._cos_anneal_ratio = 1.0
        self.numerical_gradients_delta = 0.0001

        if self.use_grid_feature:
            assert self.spatial_distortion is not None, "spatial distortion must be provided when using grid feature"

    def initialize_geo_layers(self) -> None:
        """
        Initialize layers for geometric network (sdf)
        """
        # MLP with geometric initialization
        dims = [self.config.hidden_dim for _ in range(self.config.num_layers)]
        in_dim = 3 + self.position_encoding.get_out_dim() + self.encoding.n_output_dims
        dims = [in_dim] + dims + [1 + self.config.geo_feat_dim]
        self.num_layers = len(dims)
        self.skip_in = [4]

        for layer in range(0, self.num_layers - 1):
            if layer + 1 in self.skip_in:
                out_dim = dims[layer + 1] - dims[0]
            else:
                out_dim = dims[layer + 1]

            lin = nn.Linear(dims[layer], out_dim)

            if self.config.geometric_init:
                if layer == self.num_layers - 2:
                    if not self.config.inside_outside:
                        torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[layer]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, -self.config.bias)
                    else:
                        torch.nn.init.normal_(lin.weight, mean=-np.sqrt(np.pi) / np.sqrt(dims[layer]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, self.config.bias)
                elif layer == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif layer in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3) :], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if self.config.weight_norm:
                lin = nn.utils.weight_norm(lin)
            setattr(self, "glin" + str(layer), lin)

    def set_cos_anneal_ratio(self, anneal: float) -> None:
        """Set the anneal value for the proposal network."""
        self._cos_anneal_ratio = anneal

    def update_mask(self, level: int):
        self.hash_encoding_mask[:] = 1.0
        self.hash_encoding_mask[level * self.features_per_level:] = 0

    def forward_geonetwork(self, inputs: Float[Tensor, "*batch 3"]) -> Float[Tensor, "*batch geo_features+1"]:
        """forward the geonetwork"""
        if self.use_grid_feature:
            assert self.spatial_distortion is not None, "spatial distortion must be provided when using grid feature"
            positions = self.spatial_distortion(inputs)
            # map range [-2, 2] to [0, 1]
            positions = (positions + 2.0) / 4.0
            feature = self.encoding(positions)
        else:
            feature = torch.zeros_like(inputs[:, :1].repeat(1, self.encoding.n_output_dims))

        pe = self.position_encoding(inputs)
        if not self.config.use_position_encoding:
            pe = torch.zeros_like(pe)

        inputs = torch.cat((inputs, pe, feature), dim=-1)

        # Pass through layers
        outputs = inputs

        for layer in range(0, self.num_layers - 1):
            lin = getattr(self, "glin" + str(layer))

            if layer in self.skip_in:
                outputs = torch.cat([outputs, inputs], 1) / np.sqrt(2)

            outputs = lin(outputs)

            if layer < self.num_layers - 2:
                outputs = self.softplus(outputs)
        return outputs

    # TODO: fix ... in shape annotations.
    def get_sdf(self, ray_samples: RaySamples) -> Float[Tensor, "num_samples ... 1"]:
        """predict the sdf value for ray samples"""
        positions = ray_samples.frustums.get_start_positions()
        positions_flat = positions.view(-1, 3)
        hidden_output = self.forward_geonetwork(positions_flat).view(*ray_samples.frustums.shape, -1)
        sdf, _ = torch.split(hidden_output, [1, self.config.geo_feat_dim], dim=-1)
        return sdf

    def set_numerical_gradients_delta(self, delta: float) -> None:
        """Set the delta value for numerical gradient."""
        self.numerical_gradients_delta = delta

    def gradient(self, x, skip_spatial_distortion=False, return_sdf=False):
        """compute the gradient of the ray"""
        if self.spatial_distortion is not None and not skip_spatial_distortion:
            x = self.spatial_distortion(x)

        # compute gradient in contracted space
        if self.config.use_numerical_gradients:
            # https://github.com/bennyguo/instant-nsr-pl/blob/main/models/geometry.py#L173
            delta = self.numerical_gradients_delta
            points = torch.stack(
                [
                    x + torch.as_tensor([delta, 0.0, 0.0]).to(x),
                    x + torch.as_tensor([-delta, 0.0, 0.0]).to(x),
                    x + torch.as_tensor([0.0, delta, 0.0]).to(x),
                    x + torch.as_tensor([0.0, -delta, 0.0]).to(x),
                    x + torch.as_tensor([0.0, 0.0, delta]).to(x),
                    x + torch.as_tensor([0.0, 0.0, -delta]).to(x),
                ],
                dim=0,
            )

            points_sdf = self.forward_geonetwork(points.view(-1, 3))[..., 0].view(6, *x.shape[:-1])
            gradients = torch.stack(
                [
                    0.5 * (points_sdf[0] - points_sdf[1]) / delta,
                    0.5 * (points_sdf[2] - points_sdf[3]) / delta,
                    0.5 * (points_sdf[4] - points_sdf[5]) / delta,
                ],
                dim=-1,
            )
        else:
            x.requires_grad_(True)

            y = self.forward_geonetwork(x)[:, :1]
            d_output = torch.ones_like(y, requires_grad=False, device=y.device)
            gradients = torch.autograd.grad(
                outputs=y, inputs=x, grad_outputs=d_output, create_graph=True, retain_graph=True, only_inputs=True
            )[0]
        if not return_sdf:
            return gradients
        else:
            return gradients, points_sdf

    def get_alpha(
        self,
        ray_samples: RaySamples,
        sdf: Optional[Float[Tensor, "num_samples ... 1"]] = None,
        gradients: Optional[Float[Tensor, "num_samples ... 1"]] = None,
    ) -> Float[Tensor, "num_samples ... 1"]:
        """compute alpha from sdf as in NeuS"""
        if sdf is None or gradients is None:
            inputs = ray_samples.frustums.get_start_positions()
            inputs.requires_grad_(True)
            with torch.enable_grad():
                hidden_output = self.forward_geonetwork(inputs)
                sdf, _ = torch.split(hidden_output, [1, self.config.geo_feat_dim], dim=-1)
            d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
            gradients = torch.autograd.grad(
                outputs=sdf,
                inputs=inputs,
                grad_outputs=d_output,
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]

        inv_s = self.deviation_network.get_variance()  # Single parameter

        true_cos = (ray_samples.frustums.directions * gradients).sum(-1, keepdim=True)

        # anneal as NeuS
        cos_anneal_ratio = self._cos_anneal_ratio

        # "cos_anneal_ratio" grows from 0 to 1 in the beginning training iterations. The anneal strategy below makes
        # the cos value "not dead" at the beginning training iterations, for better convergence.
        iter_cos = -(
            F.relu(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio) + F.relu(-true_cos) * cos_anneal_ratio
        )  # always non-positive

        # Estimate signed distances at section points
        estimated_next_sdf = sdf + iter_cos * ray_samples.deltas * 0.5
        estimated_prev_sdf = sdf - iter_cos * ray_samples.deltas * 0.5

        prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
        next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

        p = prev_cdf - next_cdf
        c = prev_cdf

        alpha = ((p + 1e-5) / (c + 1e-5)).clip(0.0, 1.0)

        return alpha

    def get_density(self, ray_samples: RaySamples):
        raise NotImplementedError

    def get_occupancy(self, sdf):
        """compute occupancy as in UniSurf"""
        occupancy = self.sigmoid(-10.0 * sdf)
        return occupancy

    def get_colors(
        self,
        points: Float[Tensor, "*batch 3"],
        directions: Float[Tensor, "*batch 3"],
        gradients: Float[Tensor, "*batch 3"],
        geo_features: Float[Tensor, "*batch geo_feat_dim"],
        camera_indices: Tensor,
    ) -> Float[Tensor, "*batch 3"]:
        """compute colors"""

        # diffuse color and specular tint
        if self.config.use_diffuse_color:
            raw_rgb_diffuse = self.diffuse_color_pred(geo_features.view(-1, self.config.geo_feat_dim))
        if self.config.use_specular_tint:
            tint = self.sigmoid(self.specular_tint_pred(geo_features.view(-1, self.config.geo_feat_dim)))

        normals = F.normalize(gradients, p=2, dim=-1)

        if self.config.use_reflections:
            # https://github.com/google-research/multinerf/blob/5d4c82831a9b94a87efada2eee6a993d530c4226/internal/ref_utils.py#L22
            refdirs = 2.0 * torch.sum(normals * -directions, axis=-1, keepdims=True) * normals + directions
            d = self.direction_encoding(refdirs)
        else:
            d = self.direction_encoding(directions)

        # appearance
        if self.training:
            embedded_appearance = self.embedding_appearance(camera_indices)
            # set it to zero if don't use it
            if not self.config.use_appearance_embedding:
                embedded_appearance = torch.zeros_like(embedded_appearance)
        else:
            if self.use_average_appearance_embedding:
                embedded_appearance = torch.ones(
                    (*directions.shape[:-1], self.config.appearance_embedding_dim), device=directions.device
                ) * self.embedding_appearance.mean(dim=0)
            else:
                embedded_appearance = torch.zeros(
                    (*directions.shape[:-1], self.config.appearance_embedding_dim), device=directions.device
                )
        if self.config.use_diffuse_color:
            hidden_input = [
                d,
                geo_features.view(-1, self.config.geo_feat_dim),
                embedded_appearance.view(-1, self.config.appearance_embedding_dim),
            ]
        else:
            hidden_input = [
                points,
                d,
                gradients,
                geo_features.view(-1, self.config.geo_feat_dim),
                embedded_appearance.view(-1, self.config.appearance_embedding_dim),
            ]

        if self.config.use_n_dot_v:
            n_dot_v = torch.sum(normals * directions, dim=-1, keepdims=True)
            hidden_input.append(n_dot_v)

        hidden_input = torch.cat(hidden_input, dim=-1)

        for layer in range(0, self.num_layers_color - 1):
            lin = getattr(self, "clin" + str(layer))

            hidden_input = lin(hidden_input)

            if layer < self.num_layers_color - 2:
                hidden_input = self.relu(hidden_input)

        rgb = self.sigmoid(hidden_input)

        if self.config.use_diffuse_color:
            # Initialize linear diffuse color around 0.25, so that the combined
            # linear color is initialized around 0.5.
            diffuse_linear = self.sigmoid(raw_rgb_diffuse - math.log(3.0))
            if self.config.use_specular_tint:
                specular_linear = tint * rgb
            else:
                specular_linear = 0.5 * rgb

            # TODO linear to srgb?
            # Combine specular and diffuse components and tone map to sRGB.
            rgb = torch.clamp(specular_linear + diffuse_linear, 0.0, 1.0)

        # Apply padding, mapping color to [-rgb_padding, 1+rgb_padding].
        rgb = rgb * (1 + 2 * self.config.rgb_padding) - self.config.rgb_padding

        return rgb

    def get_outputs(
        self,
        ray_samples: RaySamples,
        density_embedding: Optional[Tensor] = None,
        return_alphas: bool = False,
        return_occupancy: bool = False,
    ) -> Dict[FieldHeadNames, Tensor]:
        """compute output of ray samples"""
        if ray_samples.camera_indices is None:
            raise AttributeError("Camera indices are not provided.")

        outputs = {}

        camera_indices = ray_samples.camera_indices.squeeze()

        inputs = ray_samples.frustums.get_start_positions()
        inputs = inputs.view(-1, 3)

        directions = ray_samples.frustums.directions
        directions_flat = directions.reshape(-1, 3)

        if self.spatial_distortion is not None:
            inputs = self.spatial_distortion(inputs)
        points_norm = inputs.norm(dim=-1)
        # compute gradient in contracted space
        inputs.requires_grad_(True)
        with torch.enable_grad():
            hidden_output = self.forward_geonetwork(inputs)
            sdf, geo_feature = torch.split(hidden_output, [1, self.config.geo_feat_dim], dim=-1)

        if self.config.use_numerical_gradients:
            gradients, sampled_sdf = self.gradient(
                inputs,
                skip_spatial_distortion=True,
                return_sdf=True,
            )
            sampled_sdf = sampled_sdf.view(-1, *ray_samples.frustums.directions.shape[:-1]).permute(1, 2, 0).contiguous()
        else:
            d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
            gradients = torch.autograd.grad(
                outputs=sdf,
                inputs=inputs,
                grad_outputs=d_output,
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]
            sampled_sdf = None

        rgb = self.get_colors(inputs, directions_flat, gradients, geo_feature, camera_indices)

        density = self.laplace_density(sdf)

        rgb = rgb.view(*ray_samples.frustums.directions.shape[:-1], -1)
        sdf = sdf.view(*ray_samples.frustums.directions.shape[:-1], -1)
        density = density.view(*ray_samples.frustums.directions.shape[:-1], -1)
        gradients = gradients.view(*ray_samples.frustums.directions.shape[:-1], -1)
        normals = F.normalize(gradients, p=2, dim=-1)
        points_norm = points_norm.view(*ray_samples.frustums.directions.shape[:-1], -1)

        outputs.update(
            {
                FieldHeadNames.RGB: rgb,
                FieldHeadNames.DENSITY: density,
                FieldHeadNames.SDF: sdf,
                FieldHeadNames.NORMAL: normals,
                FieldHeadNames.GRADIENT: gradients,
                "points_norm": points_norm,
                "sampled_sdf": sampled_sdf,
            }
        )

        if return_alphas:
            alphas = self.get_alpha(ray_samples, sdf, gradients)
            outputs.update({FieldHeadNames.ALPHA: alphas})

        if return_occupancy:
            occupancy = self.get_occupancy(sdf)
            outputs.update({FieldHeadNames.OCCUPANCY: occupancy})

        return outputs

    def forward(
        self, ray_samples: RaySamples, compute_normals: bool = False, return_alphas: bool = False,
            return_occupancy: bool = False,
    ) -> Dict[FieldHeadNames, Tensor]:
        """Evaluates the field at points along the ray.

        Args:
            ray_samples: Samples to evaluate field on.
            compute normals: not currently used in this implementation.
            return_alphas: Whether to return alpha values
            return_occupancy: Whether to return occupancy values
        """
        field_outputs = self.get_outputs(ray_samples, return_alphas=return_alphas, return_occupancy=return_occupancy)
        return field_outputs
