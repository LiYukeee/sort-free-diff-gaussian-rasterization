#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from typing import NamedTuple
import torch.nn as nn
import torch
from . import _C


def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)


def rasterize_gaussians(
        sigma,
        weight_background,
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacity,
        vi,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
        if_depth_correct
):
    return _RasterizeGaussians.apply(
        sigma,
        weight_background,
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacity,
        vi,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
        if_depth_correct
    )


class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
            ctx,
            sigma,
            weight_background,
            means3D,
            means2D,
            sh,
            colors_precomp,
            opacity,
            vi,
            scales,
            rotations,
            cov3Ds_precomp,
            raster_settings,
            if_depth_correct
    ):

        # Restructure arguments the way that the C++ lib expects them
        args = (
            float(sigma[0]),
            float(weight_background[0]),
            raster_settings.bg,
            means3D,
            colors_precomp,
            opacity,
            vi,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            sh,
            raster_settings.sh_degree,
            raster_settings.campos,
            raster_settings.prefiltered,
            raster_settings.debug,
            if_depth_correct
        )

        # Invoke C++/CUDA rasterizer
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args)  # Copy them before they can be corrupted
            try:
                num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer = _C.rasterize_gaussians(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_fw.dump")
                print("\nAn error occured in forward. Please forward snapshot_fw.dump for debugging.")
                raise ex
        else:
            num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer = _C.rasterize_gaussians(*args)

        # Keep relevant tensors for backward
        ctx.num_rendered = num_rendered
        ctx.raster_settings = raster_settings
        ctx.save_for_backward(opacity, vi, torch.tensor(if_depth_correct, dtype=torch.bool), sigma, color, colors_precomp, means3D, scales, rotations,
                                cov3Ds_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer)
        # accum_weights_ptr, accum_weights_count, accum_weights_count
        # Only calcuate for sampling
        return color, radii

    @staticmethod
    def backward(ctx, grad_out_color, grad_radii):

        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        opacity, vi, if_depth_correct, sigma, render_image, colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer = ctx.saved_tensors

        # Restructure args as C++ method expects them
        args = (
            float(sigma[0]),
            render_image,
            opacity,
            vi,
            raster_settings.bg,
            means3D,
            radii,
            colors_precomp,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            grad_out_color,
            sh,
            raster_settings.sh_degree,
            raster_settings.campos,
            geomBuffer,
            num_rendered,
            binningBuffer,
            imgBuffer,
            raster_settings.debug,
            if_depth_correct
            )

        # Compute gradients for relevant tensors by invoking backward method
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args)  # Copy them before they can be corrupted
            try:
                grad_means2D, grad_colors_precomp, grad_opacity, grad_vi, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations, grad_sigma, grad_weight_background = _C.rasterize_gaussians_backward(
                    *args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_bw.dump")
                print("\nAn error occured in backward. Writing snapshot_bw.dump for debugging.\n")
                raise ex
        else:
            grad_means2D, grad_colors_precomp, grad_opacity, grad_vi, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations, grad_sigma, grad_weight_background = _C.rasterize_gaussians_backward(
                *args)

        if torch.any(torch.isnan(grad_means2D)):
            cpu_args = cpu_deep_copy_tuple(args)
            torch.save(cpu_args, "snapshot_bw.dump")
            raise ValueError("发现 grad_means2D 中有 NaN 值，程序将终止")

        grads = (
            torch.tensor([grad_sigma.mean()]).to('cuda'),
            torch.tensor([grad_weight_background.mean()]).to('cuda'),
            grad_means3D,
            grad_means2D,
            grad_sh,
            grad_colors_precomp,
            grad_opacity,
            grad_vi,
            grad_scales,
            grad_rotations,
            grad_cov3Ds_precomp,
            None,
            None
        )
        return grads


class GaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int
    tanfovx: float
    tanfovy: float
    bg: torch.Tensor
    scale_modifier: float
    viewmatrix: torch.Tensor
    projmatrix: torch.Tensor
    sh_degree: int
    campos: torch.Tensor
    prefiltered: bool
    debug: bool


class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    def markVisible(self, positions):
        # Mark visible points (based on frustum culling for camera) with a boolean
        with torch.no_grad():
            raster_settings = self.raster_settings
            visible = _C.mark_visible(
                positions,
                raster_settings.viewmatrix,
                raster_settings.projmatrix)

        return visible

    def forward(
        self,
        sigma,
        weight_background, 
        means3D,
        means2D,
        opacities,
        vi,
        shs=None,
        colors_precomp=None,
        scales=None,
        rotations=None,
        cov3D_precomp=None,
        if_depth_correct=True
    ):

        raster_settings = self.raster_settings

        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise Exception('Please provide excatly one of either SHs or precomputed colors!')

        if ((scales is None or rotations is None) and cov3D_precomp is None) or (
                (scales is not None or rotations is not None) and cov3D_precomp is not None):
            raise Exception('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')

        if shs is None:
            shs = torch.Tensor([])
        if colors_precomp is None:
            colors_precomp = torch.Tensor([])

        if scales is None:
            scales = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])

        # Invoke C++/CUDA rasterization routine
        return rasterize_gaussians(
            sigma,
            weight_background,
            means3D,
            means2D,
            shs,
            colors_precomp,
            opacities,
            vi,
            scales,
            rotations,
            cov3D_precomp,
            raster_settings,
            if_depth_correct
        )

    def visible_filter(self, means3D, scales=None, rotations=None, cov3D_precomp=None):

        raster_settings = self.raster_settings

        if scales is None:
            scales = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])

        # Invoke C++/CUDA rasterization routine
        with torch.no_grad():
            radii = _C.rasterize_aussians_filter(
                means3D,
                scales,
                rotations,
                raster_settings.scale_modifier,
                cov3D_precomp,
                raster_settings.viewmatrix,
                raster_settings.projmatrix,
                raster_settings.tanfovx,
                raster_settings.tanfovy,
                raster_settings.image_height,
                raster_settings.image_width,
                raster_settings.prefiltered,
                raster_settings.debug
            )
        return radii

