import torch
import torch.nn.functional as F
import numpy as np
import cv2
import warnings
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple, Union
from .utils_ import img_from_canvas, convert_overlay_heatmap


def draw_featmap(featmap: torch.Tensor,
                 overlaid_image: Optional[np.ndarray] = None,
                 channel_reduction: Optional[str] = "squeeze_mean",
                 topk: int = 20,
                 arrangement: Tuple[int, int] = [4, 5],
                 resize_shape: Optional[tuple] = None,
                 alpha: float = 0.2
                 ) -> np.ndarray:
    """Draw featmap.

    - If `overlaid_image` is not None, the final output image will be the
      weighted sum of img and featmap.

    - If `resize_shape` is specified, `featmap` and `overlaid_image`
      are interpolated.

    - If `resize_shape` is None and `overlaid_image` is not None,
      the feature map will be interpolated to the spatial size of the image
      in the case where the spatial dimensions of `overlaid_image` and
      `featmap` are different.

    - If `channel_reduction` is "squeeze_mean" and "select_max",
      it will compress featmap to single channel image and weighted
      sum to `overlaid_image`.

    - If `channel_reduction` is None

      - If topk <= 0, featmap is assert to be one or three
        channel and treated as image and will be weighted sum
        to ``overlaid_image``.
      - If topk > 0, it will select topk channel to show by the sum of
        each channel. At the same time, you can specify the `arrangement`
        to set the window layout.

    Args:
        featmap (torch.Tensor): The featmap to draw which format is
            (C, H, W).
        overlaid_image (np.ndarray, optional): The overlaid image.
            Defaults to None.
        channel_reduction (str, optional): Reduce multiple channels to a
            single channel. The optional value is 'squeeze_mean'
            or 'select_max'. Defaults to 'squeeze_mean'.
        topk (int): If channel_reduction is not None and topk > 0,
            it will select topk channel to show by the sum of each channel.
            if topk <= 0, tensor_chw is assert to be one or three.
            Defaults to 20.
        arrangement (Tuple[int, int]): The arrangement of featmap when
            channel_reduction is not None and topk > 0. Defaults to (4, 5).
        resize_shape (tuple, optional): The shape to scale the feature map.
            Defaults to None.
        alpha (Union[int, List[int]]): The transparency of featmap.
            Defaults to 0.5.

    Returns:
        np.ndarray: RGB image.
    """
    import matplotlib.pyplot as plt
    # check input
    # featmap: type, device, ndim
    assert isinstance(featmap, torch.Tensor), (f'`featmap` should be torch.Tensor,'
                                               f'but got {type(featmap)}')
    assert featmap.ndim == 3, f'Input dimension must be 3, ' \
                              f'but got {featmap.ndim}'
    featmap = featmap.detach().cpu()

    # overlaid_image: ndim, shape (resize_shape)
    if overlaid_image is not None:
        if overlaid_image.ndim == 2:
            overlaid_image = cv2.cvtColor(overlaid_image, cv2.COLOR_GRAY2RGB)

        if overlaid_image.shape[:2] != featmap.shape[1:]:
            warnings.warn(f"Spatial dimension of "
                          f"overlaid_image {overlaid_image.shape[:2]} and "
                          f"featmap {featmap.shape[1:]} is different")
            if resize_shape is None:
                featmap = F.interpolate(featmap[None],
                                        overlaid_image.shape[:2],
                                        mode="bilinear",
                                        align_corners=False)[0]

    # resize_shape: shape
    if resize_shape is not None:
        featmap = F.interpolate(featmap[None],
                                resize_shape,
                                mode="bilinear",
                                align_corners=False)[0]

        if overlaid_image is not None:
            overlaid_image = cv2.resize(overlaid_image, resize_shape[::-1])

    # visual strage
    # channel reduction strage
    if channel_reduction is not None:
        assert channel_reduction in ["squeeze_mean", "select_max"], \
            f'mode only support "squeeze_mean and select_max' \
            f'but got {channel_reduction}'
        if channel_reduction == "select_max":
            sum_channel_featmap = torch.sum(featmap, dim=(1, 2))
            _, indices = torch.topk(sum_channel_featmap, 1)
            feat_map = featmap[indices]
        else:
            feat_map = torch.mean(featmap, dim=0)

        return convert_overlay_heatmap(feat_map, overlaid_image, alpha)
    # topk <= 0: visual direct
    elif topk <= 0:
        featmap_channel = featmap.shape[0]

        assert featmap_channel in [1, 3], f"input tensor dimension must be 1 or 3" \
                                          f"when topk is less than 1," \
                                          f"but got {featmap_channel}"
        return convert_overlay_heatmap(featmap, overlaid_image, alpha)
    # topk > 0
    else:
        row, col = arrangement
        channel, height, width = featmap.shape
        assert row * col >= topk, f"topk must less than the product of row and col of arrangement" \
                                  f"the arrangement is {arrangement}," \
                                  f"while the topk is {topk}"

        topk = min(channel, topk)
        sum_channel_featmap = torch.sum(featmap, dim=(1, 2))
        _, indices = torch.topk(sum_channel_featmap, topk)
        topk_featmap = featmap[indices]

        # plt figure
        fig = plt.figure(frameon=False)

        # set window layout
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
        dpi = fig.get_dpi()
        fig.set_size_inches((width * col + 1e-2) / dpi, (height * row + 1e-2) / dpi)
        for i in range(topk):
            axes = fig.add_subplot(row, col, i+1)
            axes.axis('off')
            axes.text(2, 15, f"channel: {indices[i]}", fontsize=10)
            axes.imshow(convert_overlay_heatmap(topk_featmap[i], overlaid_image, alpha))

        image = img_from_canvas(fig.canvas)
        plt.close()
        return image
