# Illumination-based Transformations Improve Skin Lesion Segmentation in Dermoscopic Images

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This is the code corresponding to our CVPR ISIC 2020 paper. If you use our code, please cite our paper: 

Kumar Abhishek, Ghassan Hamarneh, Mark S. Drew, "[Illumination-based Transformations Improve Skin Lesion Segmentation in Dermoscopic Images](https://arxiv.org/abs/2003.10111)", The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) ISIC Skin Image Analysis Workshop, 2020.

The corresponding bibtex entry is:

```
@InProceedings{Abhishek_2020_CVPRW,
author = {Abhishek, Kumar and Hamarneh, Ghassan and Drew, Mark S.},
title = {Illumination-based Transformations Improve Skin Lesion Segmentation in Dermoscopic Images},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) ISIC Skin Image Analysis Workshop},
month = {June},
year = {2020}
}
```

## Dependencies
- PyTorch
- NumPy
- PIL
- SciPy (==1.1.0; for `imread` and `imresize`)

We also need to convert between the RGB and HSV color spaces, and `scikit-image` provides this functionality. However, `scikit-image` has issues with NumPy<sup>[1](https://github.com/numpy/numpy/issues/12744),[2](https://github.com/scikit-image/scikit-image/issues/3586),[3](https://github.com/scikit-image/scikit-image/issues/3906)</sup>, and therefore to avoid installing it, we have borrowed the necessary functions from `scikit-image` [source code](https://github.com/scikit-image/scikit-image/tree/dfb7a0922a4d4cc4841c74c1657086ca5d8a6b35) and put them in `skimg_local.py`.

## Usage

An example usage is shown in `Transformations.ipynb`, and three sample skin lesion images from the [ISIC Archive](https://www.isic-archive.com/) are included in the `images` directory.