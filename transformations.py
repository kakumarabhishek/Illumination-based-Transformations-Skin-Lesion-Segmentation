from utils import *
from skimg_local import rgb2hsv, hsv2rgb


def select_channels(img_RGB):
    """
    Returns the R' and V* channels for a skin lesion image.

    Args:
        img_RGB (np.array): The RGB image of the skin lesion
    """
    img_RGB_norm = img_RGB / 255.0
    img_r_norm = img_RGB_norm[..., 0] / (
        img_RGB_norm[..., 0] + img_RGB_norm[..., 1] + img_RGB_norm[..., 2]
    )
    img_v = np.max(img_RGB, axis=2)

    return (img_r_norm, img_v)


def calculate_GRAY(img_RGB):
    """
    Returns the single channel grayscale representation of
    the skin lesion.

    Args:
        img_RGB (np.array): The RGB image of the skin lesion
    """
    img_torch = torch.from_numpy(img_RGB) + eps

    X = torch.log(torch.reshape(img_torch, (-1, 3)))
    X_mean = torch.mean(X, 0)
    X -= X_mean.expand_as(X)

    U, S, V = torch.svd(torch.t(X))
    C = torch.mm(X, U[..., :1])
    C_reshaped = torch.reshape(C, (128, 128, -1))[..., 0]

    C_reshaped_np = C_reshaped.cpu().detach().numpy()

    return C_reshaped_np


def calculate_Intrinsic_SA(img_RGB):
    """
    Returns the illumination invariant 'intrinsic' image and 
    the shading attentuated representation for the skin lesion.

    Args:
        img_RGB (np.array): The RGB image of the skin lesion
    """
    img_torch = torch.from_numpy(img_RGB) + eps
    angle, projected = entropy_intrinsic(img_torch, calculate_intrinsic_img=True)
    projected_np = projected.cpu().detach().numpy()

    img_RGB_norm = img_RGB / 255.0
    projected_norm = projected_np / 255.0
    img_HSV = rgb2hsv(img_RGB)
    matched = hist_match(img_HSV[..., 2], projected_norm)
    img_HSV[..., 2] = matched
    img_RGB_SA_norm = hsv2rgb(img_HSV)
    img_RGB_SA = img_RGB_SA_norm * 255

    return (projected_np, img_RGB_SA)
