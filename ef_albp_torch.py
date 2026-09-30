"""Core EF-ALBP operations used by Mamba-EdgeNet."""

import torch
import torch.nn.functional as F


_CACHE = {}
ALBP_WEIGHT_SUM = float(sum(2 ** a for a in range(11)))


def edge_aware_filtering_torch(image, sigma_s=3, sigma_r=1, tau=0.03):
    """Apply differentiable edge-aware filtering to a grayscale image tensor.

    Args:
        image: Tensor of shape [B, 1, H, W].
        sigma_s: Spatial weighting parameter.
        sigma_r: Range weighting parameter.
        tau: Soft threshold parameter.

    Returns:
        Filtered tensor with the same shape as ``image``.
    """
    _, _, h, w = image.shape

    sigma_s2 = 2 * sigma_s ** 2
    w0 = torch.exp(-2 / sigma_s2)
    w1 = torch.exp(-1 / sigma_s2)
    w_center = torch.ones_like(w0)

    spatial_weights = torch.stack(
        [w0, w1, w0, w1, w_center, w1, w0, w1, w0]
    ).view(1, 1, 9, 1, 1)

    padded = F.pad(image, (1, 1, 1, 1), mode="reflect")
    unfolded = F.unfold(padded, kernel_size=3)
    unfolded = unfolded.view(image.shape[0], 1, 9, h, w)

    diff = unfolded - image.unsqueeze(2)
    range_weights = torch.exp(-(diff ** 2) / (2 * sigma_r ** 2))

    condition = torch.sigmoid((tau - diff.abs()) * 2.0)
    spatial_alt = (2 / (3 * torch.pi * sigma_s ** 2)) * spatial_weights
    combined_spatial = (
        condition * spatial_weights
        + (1 - condition) * spatial_alt
    )

    weights = combined_spatial * range_weights
    numerator = (diff * weights).sum(dim=2)
    denominator = weights.sum(dim=2) + 1e-8

    return image + numerator / denominator


def _get_powers_tensor(device, dtype):
    """Return cached directional binary weights."""
    key = ("powers", device, dtype)
    if key not in _CACHE:
        _CACHE[key] = torch.tensor(
            [2.0 ** i for i in range(8)],
            device=device,
            dtype=dtype,
        ).view(1, 1, 8, 1, 1)
    return _CACHE[key]


def albp_torch(filtered_img, alphaT=5):
    """Compute the differentiable ALBP response.

    Args:
        filtered_img: Tensor of shape [B, C, H, W].
        alphaT: Learnable ALBP threshold parameter.

    Returns:
        ALBP response tensor with shape [B, C, H, W].
    """
    directions = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, 1),
        (1, 1), (1, 0), (1, -1),
        (0, -1),
    ]

    diffs = torch.stack(
        [
            torch.roll(filtered_img, shifts=(dy, dx), dims=(2, 3))
            - filtered_img
            for dy, dx in directions
        ],
        dim=2,
    )

    diffs_scaled = torch.tanh(diffs * 255.0 / 10.0)
    mask = torch.sigmoid((diffs * 255.0 - alphaT) * 10.0)

    response = ALBP_WEIGHT_SUM * (diffs_scaled ** 2) * mask
    powers = _get_powers_tensor(filtered_img.device, filtered_img.dtype)
    accumulated = (response * powers).sum(dim=2)

    return torch.sigmoid(accumulated * 0.1)


def ef_albp_torch(image):
    """Apply EF followed by ALBP with the default parameters."""
    filtered = edge_aware_filtering_torch(image)
    return albp_torch(filtered)


def ef_albp(image, sigma_s=8, sigma_r=0.7, tau=0.13, alphaT=6):
    """Apply EF-ALBP with explicitly supplied parameters."""
    filtered = edge_aware_filtering_torch(
        image,
        sigma_s=sigma_s,
        sigma_r=sigma_r,
        tau=tau,
    )
    return albp_torch(filtered, alphaT=alphaT)
