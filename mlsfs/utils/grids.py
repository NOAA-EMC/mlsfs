import numpy as np
import torch

from torch_harmonics.quadrature import legendre_gauss_weights, clenshaw_curtiss_weights
class GridQuadrature(torch.nn.Module):
    def __init__(self, quadrature_rule, img_shape, crop_shape=None, crop_offset=(0, 0), normalize=False, pole_mask=None):
        super(GridQuadrature, self).__init__()

        if quadrature_rule == "naive":
            jacobian = torch.clamp(torch.sin(torch.linspace(0, torch.pi, img_shape[0])), min=0.0)
            dtheta = torch.pi / img_shape[0]
            dlambda = 2 * torch.pi / img_shape[1]
            dA = dlambda * dtheta
            quad_weight = dA * jacobian.unsqueeze(1)
            quad_weight = quad_weight.tile(1, img_shape[1])
            # numerical precision can be an issue here, make sure it sums to 4pi:
            quad_weight = quad_weight * (4.0 * torch.pi) / torch.sum(quad_weight)
        elif quadrature_rule == "clenshaw-curtiss":
            cost, w = clenshaw_curtiss_weights(img_shape[0], -1, 1)
            weights = torch.from_numpy(w)
            dlambda = 2 * torch.pi / img_shape[1]
            quad_weight = dlambda * torch.from_numpy(w).unsqueeze(1)
            quad_weight = quad_weight.tile(1, img_shape[1])
        elif quadrature_rule == "legendre-gauss":
            cost, w = legendre_gauss_weights(img_shape[0], -1, 1)
            weights = torch.from_numpy(w)
            dlambda = 2 * torch.pi / img_shape[1]
            quad_weight = dlambda * torch.from_numpy(w).unsqueeze(1)
            quad_weight = quad_weight.tile(1, img_shape[1])
        else:
            raise ValueError(f"Unknown quadrature rule {quadrature_rule}")

        # apply normalization
        if normalize:
            quad_weight = quad_weight / (4.0 * torch.pi)

        # apply pole mask
        if (pole_mask is not None) and (pole_mask > 0):
            quad_weight[:pole_mask, :] = 0.0
            quad_weight[sizes[0] - pole_mask :, :] = 0.0

        # crop globally if requested
        if crop_shape is not None:
            quad_weight = quad_weight[crop_offset[0] : crop_offset[0] + crop_shape[0], crop_offset[1] : crop_offset[1] + crop_shape[1]]

        # make it contiguous
        quad_weight = quad_weight.contiguous()

        # reshape
        H, W = quad_weight.shape
        quad_weight = quad_weight.reshape(1, 1, H, W)

        self.register_buffer("quad_weight", quad_weight)

    def forward(self, x):
        # integrate over last two axes only:
        return torch.sum(x * self.quad_weight, dim=(-2, -1))
