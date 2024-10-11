import torch

def l2loss_sphere(quadrature, prd, tar, chw=None, relative=False, squared=True):
    num_examples = prd.size()[0]

    loss = quadrature(torch.abs(prd - tar)**2)
    loss = loss.reshape(num_examples, -1)

    if relative:
        tar_norms = quadrature(torch.abs(tar)**2)
        tar_norms = tar_norms.reshape(num_examples, -1)

        loss = loss / tar_norms

    if not squared:
        loss = torch.sqrt(loss)

    #apply channel weighting
    if chw is not None:
        loss = chw * loss

    return torch.sum(loss)

def spectral_l2loss_sphere(solver, prd, tar, relative=False, squared=True):
    # compute coefficients
    coeffs = torch.view_as_real(solver.sht(prd - tar))
    coeffs = coeffs[..., 0]**2 + coeffs[..., 1]**2
    norm2 = coeffs[..., :, 0] + 2 * torch.sum(coeffs[..., :, 1:], dim=-1)
    loss = torch.sum(norm2, dim=(-1,-2))

    if relative:
        tar_coeffs = torch.view_as_real(solver.sht(tar))
        tar_coeffs = tar_coeffs[..., 0]**2 + tar_coeffs[..., 1]**2
        tar_norm2 = tar_coeffs[..., :, 0] + 2 * torch.sum(tar_coeffs[..., :, 1:], dim=-1)
        tar_norm2 = torch.sum(tar_norm2, dim=(-1,-2))
        loss = loss / tar_norm2

    if not squared:
        loss = torch.sqrt(loss)
    loss = loss.mean()

    return loss
