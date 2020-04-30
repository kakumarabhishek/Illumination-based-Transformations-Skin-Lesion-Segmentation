import torch
import numpy as np

eps = np.finfo(np.float32).eps

PI = 3.141592

chrom0 = torch.tensor([[1.0], [1.0], [1.0]])
chrom0 /= torch.norm(chrom0, p=2)
temp = torch.tensor([[0.0], [0.0], [1.0]])
chrom1 = torch.cross(chrom0, temp)
chrom1 /= torch.norm(chrom1, p=2)
chrom2 = torch.cross(chrom0, chrom1)
chrom2 /= torch.norm(chrom2, p=2)

theta = torch.FloatTensor([PI / 180 * (idx + 1) for idx in range(180)])
X, Y = torch.cos(theta), torch.sin(theta)
XY = torch.cat((X.reshape(1, -1), Y.reshape(1, -1)), dim=0)


def entropy_intrinsic(image, calculate_intrinsic_img=False):
    img_flat = image.reshape(-1, 3)

    log_geomean = torch.log(
        torch.pow(
            torch.mul(torch.mul(img_flat[:, 0], img_flat[:, 1]), img_flat[:, 2]),
            (1 / 3.0),
        )
    ).float()
    allrgbslog = torch.log(img_flat).float()
    allrgbslog[:, 0] -= log_geomean
    allrgbslog[:, 1] -= log_geomean
    allrgbslog[:, 2] -= log_geomean

    chi = torch.cat(
        (torch.matmul(allrgbslog, chrom1), torch.matmul(allrgbslog, chrom2)), dim=1
    )

    entropy = torch.zeros_like(theta)

    projGrey = torch.matmul(chi, XY)
    projGrey_np = projGrey.cpu().numpy()
    P05 = np.percentile(projGrey_np, 5, 0)
    P95 = np.percentile(projGrey_np, 95, 0)

    for idx, col in enumerate(projGrey_np.T):
        p05, p95 = P05[idx], P95[idx]
        im = col[np.where(np.logical_and(col > p05, col < p95))[0]]
        n = im.shape[0]
        bin_wid = 3.5 * np.power(n, (-1.0 / 3)) * np.std(im)
        his_up, his_low = np.max(im), np.min(im)
        nbin = np.int(np.ceil((his_up - his_low) / bin_wid))
        bins = np.arange(his_low, (nbin + 1) * bin_wid + his_low, bin_wid)
        hist, _ = np.histogram(im, bins)
        n_hist = hist / n

        tmp_entropy = 0
        for bin_idx in range(nbin):
            if n_hist[bin_idx] != 0:
                tmp_entropy -= n_hist[bin_idx] * np.log2(n_hist[bin_idx])

        entropy[idx] = tmp_entropy

    if calculate_intrinsic_img:
        projected = torch.matmul(chi, XY[:, int(np.argmin(tmp_entropy))]).reshape(
            128, 128
        )
        return (np.argmin(entropy), projected)
    else:
        return np.argmin(entropy)


def hist_match(source, template):
    """
    Source: https://stackoverflow.com/a/33047048
    Adjust the pixel values of a grayscale image such that its histogram
    matches that of a target image

    Arguments:
    -----------
        source: np.ndarray
            Image to transform; the histogram is computed over the flattened
            array
        template: np.ndarray
            Template image; can have different dimensions to source
    Returns:
    -----------
        matched: np.ndarray
            The transformed output image
    """

    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()

    # get the set of unique pixel values and their corresponding indices and
    # counts
    s_values, bin_idx, s_counts = np.unique(
        source, return_inverse=True, return_counts=True
    )
    t_values, t_counts = np.unique(template, return_counts=True)

    # take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution functions for the source and
    # template images (maps pixel value --> quantile)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    # interpolate linearly to find the pixel values in the template image
    # that correspond most closely to the quantiles in the source image
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

    return interp_t_values[bin_idx].reshape(oldshape)
