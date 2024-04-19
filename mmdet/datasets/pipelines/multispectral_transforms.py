import copy
import inspect

import mmcv
import numpy as np
from numpy import random

from mmdet.core import PolygonMasks
from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps
from ..builder import PIPELINES
from .transforms import Resize

_MAX_LEVEL = 10


def level_to_value(level, max_value):
    """Map from level to values based on max_value."""
    return (level / _MAX_LEVEL) * max_value


def enhance_level_to_value(level, a=1.8, b=0.1):
    """Map from level to values."""
    return (level / _MAX_LEVEL) * a + b


@PIPELINES.register_module()
class MultiNormalize:
    """Normalize the image.

    Added key is "img_norm_cfg".

    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean_list, std_list, to_rgb=True):
        self.mean = [np.array(mean, dtype=np.float32) for mean in mean_list]
        self.std = [np.array(std, dtype=np.float32) for std in std_list]
        self.to_rgb = to_rgb

    def __call__(self, results):
        """Call function to normalize images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """
        for i, key in enumerate(results.get('img_fields', ['img1', 'img2'])):
            results[key] = mmcv.imnormalize(results[key], self.mean[i], self.std[i], 
                                            self.to_rgb)
        results['img_norm_cfg'] = dict(
            mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(mean={self.mean}, std={self.std}, to_rgb={self.to_rgb})'
        return repr_str

@PIPELINES.register_module()
class MultiResize(Resize):
    """Resize images & bbox & mask.

    This transform resizes the input image to some scale. Bboxes and masks are
    then resized with the same scale factor. If the input dict contains the key
    "scale", then the scale in the input dict is used, otherwise the specified
    scale in the init method is used. If the input dict contains the key
    "scale_factor" (if MultiScaleFlipAug does not give img_scale but
    scale_factor), the actual scale will be computed by image shape and
    scale_factor.

    `img_scale` can either be a tuple (single-scale) or a list of tuple
    (multi-scale). There are 3 multiscale modes:

    - ``ratio_range is not None``: randomly sample a ratio from the ratio \
      range and multiply it with the image scale.
    - ``ratio_range is None`` and ``multiscale_mode == "range"``: randomly \
      sample a scale from the multiscale range.
    - ``ratio_range is None`` and ``multiscale_mode == "value"``: randomly \
      sample a scale from multiple scales.

    Args:
        img_scale (tuple or list[tuple]): Images scales for resizing.
        multiscale_mode (str): Either "range" or "value".
        ratio_range (tuple[float]): (min_ratio, max_ratio)
        keep_ratio (bool): Whether to keep the aspect ratio when resizing the
            image.
        bbox_clip_border (bool, optional): Whether to clip the objects outside
            the border of the image. In some dataset like MOT17, the gt bboxes
            are allowed to cross the border of images. Therefore, we don't
            need to clip the gt bboxes in these cases. Defaults to True.
        backend (str): Image resize backend, choices are 'cv2' and 'pillow'.
            These two backends generates slightly different results. Defaults
            to 'cv2'.
        override (bool, optional): Whether to override `scale` and
            `scale_factor` so as to call resize twice. Default False. If True,
            after the first resizing, the existed `scale` and `scale_factor`
            will be ignored so the second resizing can be allowed.
            This option is a work-around for multiple times of resize in DETR.
            Defaults to False.
    """

    def __init__(self,
                 img_scale=None,
                 main_scale_idx=1,
                 multiscale_mode='range',
                 ratio_range=None,
                 keep_ratio=True,
                 bbox_clip_border=True,
                 backend='cv2',
                 override=False):
        super(MultiResize, self).__init__(img_scale,
                                          multiscale_mode,
                                          ratio_range,
                                          keep_ratio,
                                          bbox_clip_border,
                                          backend,
                                          override)
        if self.img_scale is not None:
            if isinstance(img_scale[0], list):
                self.img_scale1 = img_scale[0]
            else:
                self.img_scale1 = [img_scale[0]]
            if isinstance(img_scale[1], list):
                self.img_scale2 = img_scale[1]
            else:
                self.img_scale2 = [img_scale[1]]
            assert mmcv.is_list_of(self.img_scale1, tuple)
            assert mmcv.is_list_of(self.img_scale2, tuple)
            self.main_scale_idx = main_scale_idx

    def _random_scale(self, results):
        """Randomly sample an img_scale according to ``ratio_range`` and
        ``multiscale_mode``.

        If ``ratio_range`` is specified, a ratio will be sampled and be
        multiplied with ``img_scale``.
        If multiple scales are specified by ``img_scale``, a scale will be
        sampled according to ``multiscale_mode``.
        Otherwise, single scale will be used.

        Args:
            results (dict): Result dict from :obj:`dataset`.

        Returns:
            dict: Two new keys 'scale` and 'scale_idx` are added into \
                ``results``, which would be used by subsequent pipelines.
        """

        if self.ratio_range is not None:
            scale1, scale_idx1 = self.random_sample_ratio(
                self.img_scale1[0], self.ratio_range)
            scale2, scale_idx2 = self.random_sample_ratio(
                self.img_scale2[0], self.ratio_range)
        elif len(self.img_scale1) == 1 and len(self.img_scale2) == 1:
            scale1, scale_idx1 = self.img_scale1[0], 0
            scale2, scale_idx2 = self.img_scale2[0], 0
        elif self.multiscale_mode == 'range':
            scale1, scale_idx1 = self.random_sample(self.img_scale1)
            scale2, scale_idx2 = self.random_sample(self.img_scale2)
        elif self.multiscale_mode == 'value':
            scale1, scale_idx1 = self.random_select(self.img_scale1)
            scale2, scale_idx2 = self.random_select(self.img_scale2)
        else:
            raise NotImplementedError

        results['scale1'] = scale1
        results['scale_idx1'] = scale_idx1
        results['scale2'] = scale2
        results['scale_idx2'] = scale_idx2

    def _resize_img(self, results):
        """Resize images with ``results['scale']``."""
        for key in results.get('img_fields', ['img1', 'img2']):
            if self.keep_ratio:
                img, scale_factor = mmcv.imrescale(
                    results[key],
                    results['scale'+key[-1]],
                    return_scale=True,
                    backend=self.backend)
                # the w_scale and h_scale has minor difference
                # a real fix should be done in the mmcv.imrescale in the future
                new_h, new_w = img.shape[:2]
                h, w = results[key].shape[:2]
                w_scale = new_w / w
                h_scale = new_h / h
            else:
                img, w_scale, h_scale = mmcv.imresize(
                    results[key],
                    results['scale'+key[-1]],
                    return_scale=True,
                    backend=self.backend)
            results[key] = img

            scale_factor = np.array([w_scale, h_scale, w_scale, h_scale],
                                    dtype=np.float32)
            results['img_shape'+key[-1]] = img.shape
            # in case that there is no padding
            results['pad_shape'+key[-1]] = img.shape
            results['scale_factor'+key[-1]] = scale_factor
            results['keep_ratio'+key[-1]] = self.keep_ratio
        results['img_shape'] = results['img_shape'+str(self.main_scale_idx)]
        results['pad_shape'] = results['pad_shape'+str(self.main_scale_idx)]
        results['scale_factor'] = results['scale_factor'+str(self.main_scale_idx)]
        results['keep_ratio'] = self.keep_ratio

    def _resize_bboxes(self, results):
        """Resize bounding boxes with ``results['scale_factor']``."""
        for key in results.get('bbox_fields', []):
            bboxes = results[key] * results['scale_factor'+str(self.main_scale_idx)]
            if self.bbox_clip_border:
                img_shape = results['img_shape'+str(self.main_scale_idx)]
                bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1])
                bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0])
            results[key] = bboxes
            
    def _resize_masks(self, results):
        """Resize masks with ``results['scale']``"""
        for key in results.get('mask_fields', []):
            if results[key] is None:
                continue
            if self.keep_ratio:
                results[key] = results[key].rescale(results['scale'+str(self.main_scale_idx)])
            else:
                results[key] = results[key].resize(results['img_shape'+str(self.main_scale_idx)][:2])

    def _resize_seg(self, results):
        """Resize semantic segmentation map with ``results['scale']``."""
        for key in results.get('seg_fields', []):
            if self.keep_ratio:
                gt_seg1 = mmcv.imrescale(
                    results[key],
                    results['scale'],
                    interpolation='nearest',
                    backend=self.backend)
            else:
                gt_seg1 = mmcv.imresize(
                    results[key],
                    results['scale'],
                    interpolation='nearest',
                    backend=self.backend)
            results[key] = gt_seg1

    def __call__(self, results):
        """Call function to resize images, bounding boxes, masks, semantic
        segmentation map.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Resized results, 'img_shape', 'pad_shape', 'scale_factor', \
                'keep_ratio' keys are added into result dict.
        """

        if 'scale' not in results:
            if 'scale_factor' in results:
                img_shape = results['img_shape']
                scale_factor = results['scale_factor']
                assert isinstance(scale_factor, float)
                results['scale'] = tuple(
                    [int(x * scale_factor) for x in img_shape][::-1])
            else:
                self._random_scale(results)
        else:
            if not self.override:
                assert 'scale_factor' not in results, (
                    'scale and scale_factor cannot be both set.')
            else:
                results.pop('scale')
                if 'scale_factor' in results:
                    results.pop('scale_factor')
                self._random_scale(results)

        self._resize_img(results)
        self._resize_bboxes(results)
        self._resize_masks(results)
        self._resize_seg(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(img_scale={self.img_scale}, '
        repr_str += f'multiscale_mode={self.multiscale_mode}, '
        repr_str += f'ratio_range={self.ratio_range}, '
        repr_str += f'keep_ratio={self.keep_ratio}, '
        repr_str += f'bbox_clip_border={self.bbox_clip_border})'
        return repr_str


@PIPELINES.register_module()
class RandomMasking:
    def __init__(self, p=(0.25, 0.25, 0.5)):
        # probabilities of masking RGB, T and nothing
        self.p = p

    def __call__(self, results):
        sample_mode = (0, 1, 2)
        mode = random.choice(sample_mode, p=self.p)
        if mode == 0 or mode == 1:
            key = results.get('img_fields', ['img1', 'img2'])[mode]
            results[key] = np.zeros(results[key].shape)
        return results


@PIPELINES.register_module()
class SpectralShift:
    # shift visible images
    def __init__(self, shift_ratio_range=(0, 0.05)):
        self.min_ratio, self.max_ratio = shift_ratio_range

    def __call__(self, results):
        img = results['img1']
        h, w, c = img.shape
        y_ratio = random.uniform(self.min_ratio, self.max_ratio)
        x_ratio = random.uniform(self.min_ratio, self.max_ratio)
        delta_y = int(h * y_ratio)
        delta_x = int(w * x_ratio)
        # expand and crop
        expand_img = np.full((h+2*delta_y, w+2*delta_x, c), 255, dtype=img.dtype)        
        expand_img[delta_y:delta_y+h, delta_x:delta_x+w] = img
        direction = random.choice((0, 1, 2, 3))
        if direction == 0:
            img = expand_img[:h, :w]
        elif direction == 1:
            img = expand_img[:h, 2*delta_x:]
        elif direction == 2:
            img = expand_img[2*delta_y:, 2*delta_x:]
        else:
            img = expand_img[2*delta_y:, :w]
        results['img1'] = img
        return results

@PIPELINES.register_module()
class MultiColorTransform:
    """Apply Color transformation to image. The bboxes, masks, and
    segmentations are not modified.

    Args:
        level (int | float): Should be in range [0,_MAX_LEVEL].
        prob (float): The probability for performing Color transformation.
    """

    def __init__(self, level, prob=0.5, img_idx=None):
        assert isinstance(level, (int, float)), \
            'The level must be type int or float.'
        assert 0 <= level <= _MAX_LEVEL, \
            'The level should be in range [0,_MAX_LEVEL].'
        assert 0 <= prob <= 1.0, \
            'The probability should be in range [0,1].'
        self.level = level
        self.prob = prob
        self.factor = enhance_level_to_value(level)
        self.img_idx = img_idx

    def _adjust_color_img(self, results, factor=1.0):
        """Apply Color transformation to image."""
        if self.img_idx is None:
            for key in results.get('img_fields',['img']):
                img = results[key]
                results[key] = mmcv.adjust_color(img, factor).astype(img.dtype)
        else:
            for idx in self.img_idx:
                # NOTE defaultly the image should be BGR format
                img = results['img'+str(idx)]
                results['img'+str(idx)] = mmcv.adjust_color(img, factor).astype(img.dtype)

    def __call__(self, results):
        """Call function for Color transformation.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Colored results.
        """
        if np.random.rand() > self.prob:
            return results
        self._adjust_color_img(results, self.factor)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(level={self.level}, '
        repr_str += f'prob={self.prob})'
        return repr_str

@PIPELINES.register_module()
class MultiBrightnessTransform:
    """Apply Brightness transformation to image. The bboxes, masks and
    segmentations are not modified.

    Args:
        level (int | float): Should be in range [0,_MAX_LEVEL].
        prob (float): The probability for performing Brightness transformation.
    """

    def __init__(self, level, prob=0.5, img_idx=None):
        assert isinstance(level, (int, float)), \
            'The level must be type int or float.'
        assert 0 <= level <= _MAX_LEVEL, \
            'The level should be in range [0,_MAX_LEVEL].'
        assert 0 <= prob <= 1.0, \
            'The probability should be in range [0,1].'
        self.level = level
        self.prob = prob
        self.factor = enhance_level_to_value(level)
        self.img_idx = img_idx

    def _adjust_brightness_img(self, results, factor=1.0):
        """Adjust the brightness of image."""
        if self.img_idx is None:
            for key in results.get('img_fields', ['img']):
                img = results[key]
                results[key] = mmcv.adjust_brightness(img,
                                                    factor).astype(img.dtype)
        else:
            for idx in self.img_idx:
                img = results['img'+str(idx)]
                results['img'+str(idx)] = mmcv.adjust_brightness(img,
                                                    factor).astype(img.dtype)
    def __call__(self, results):
        """Call function for Brightness transformation.

        Args:
            results (dict): Results dict from loading pipeline.

        Returns:
            dict: Results after the transformation.
        """
        if np.random.rand() > self.prob:
            return results
        self._adjust_brightness_img(results, self.factor)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(level={self.level}, '
        repr_str += f'prob={self.prob})'
        return repr_str


@PIPELINES.register_module()
class MultiContrastTransform:
    """Apply Contrast transformation to image. The bboxes, masks and
    segmentations are not modified.

    Args:
        level (int | float): Should be in range [0,_MAX_LEVEL].
        prob (float): The probability for performing Contrast transformation.
    """

    def __init__(self, level, prob=0.5, img_idx=None):
        assert isinstance(level, (int, float)), \
            'The level must be type int or float.'
        assert 0 <= level <= _MAX_LEVEL, \
            'The level should be in range [0,_MAX_LEVEL].'
        assert 0 <= prob <= 1.0, \
            'The probability should be in range [0,1].'
        self.level = level
        self.prob = prob
        self.factor = enhance_level_to_value(level)
        self.img_idx = img_idx

    def _adjust_contrast_img(self, results, factor=1.0):
        """Adjust the image contrast."""
        if self.img_idx is None:
            for key in results.get('img_fields', ['img']):
                img = results[key]
                results[key] = mmcv.adjust_contrast(img, factor).astype(img.dtype)
        else:
            for idx in self.img_idx:
                img = results['img'+str(idx)]
                results['img'+str(idx)] = mmcv.adjust_contrast(img, factor).astype(img.dtype)

    def __call__(self, results):
        """Call function for Contrast transformation.

        Args:
            results (dict): Results dict from loading pipeline.

        Returns:
            dict: Results after the transformation.
        """
        if np.random.rand() > self.prob:
            return results
        self._adjust_contrast_img(results, self.factor)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(level={self.level}, '
        repr_str += f'prob={self.prob})'
        return repr_str
