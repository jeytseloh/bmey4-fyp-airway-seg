from batchgenerators.transforms.abstract_transforms import AbstractTransform
from batchgenerators.augmentations.spatial_transformations import augment_spatial, augment_spatial_2, \
    augment_channel_translation, \
    augment_mirroring, augment_transpose_axes, augment_zoom, augment_resize, augment_rot90
from batchgenerators.augmentations.utils import create_zero_centered_coordinate_mesh, elastic_deform_coordinates, \
    interpolate_img, \
    rotate_coords_2d, rotate_coords_3d, scale_coords, resize_segmentation, resize_multichannel_image, \
    elastic_deform_coordinates_2
from batchgenerators.augmentations.crop_and_pad_augmentations import get_lbs_for_center_crop, get_lbs_for_random_crop
from batchgenerators.transforms.spatial_transforms import SpatialTransform
import numpy as np

class SpatialTransformCL(SpatialTransform):
    """The ultimate spatial transform generator. Rotation, deformation, scaling, cropping: It has all you ever dreamed
    of. Computational time scales only with patch_size, not with input patch size or type of augmentations used.
    Internally, this transform will use a coordinate grid of shape patch_size to which the transformations are
    applied (very fast). Interpolation on the image data will only be done at the very end

    Args:
        patch_size (tuple/list/ndarray of int): Output patch size

        patch_center_dist_from_border (tuple/list/ndarray of int, or int): How far should the center pixel of the
        extracted patch be from the image border? Recommended to use patch_size//2.
        This only applies when random_crop=True

        do_elastic_deform (bool): Whether or not to apply elastic deformation

        alpha (tuple of float): magnitude of the elastic deformation; randomly sampled from interval

        sigma (tuple of float): scale of the elastic deformation (small = local, large = global); randomly sampled
        from interval

        do_rotation (bool): Whether or not to apply rotation

        angle_x, angle_y, angle_z (tuple of float): angle in rad; randomly sampled from interval. Always double check
        whether axes are correct!

        do_scale (bool): Whether or not to apply scaling

        scale (tuple of float): scale range ; scale is randomly sampled from interval

        border_mode_data: How to treat border pixels in data? see scipy.ndimage.map_coordinates

        border_cval_data: If border_mode_data=constant, what value to use?

        order_data: Order of interpolation for data. see scipy.ndimage.map_coordinates

        border_mode_seg: How to treat border pixels in seg? see scipy.ndimage.map_coordinates

        border_cval_seg: If border_mode_seg=constant, what value to use?

        order_seg: Order of interpolation for seg. see scipy.ndimage.map_coordinates. Strongly recommended to use 0!
        If !=0 then you will have to round to int and also beware of interpolation artifacts if you have more then
        labels 0 and 1. (for example if you have [0, 0, 0, 2, 2, 1, 0] the neighboring [0, 0, 2] bay result in [0, 1, 2])

        random_crop: True: do a random crop of size patch_size and minimal distance to border of
        patch_center_dist_from_border. False: do a center crop of size patch_size

        independent_scale_for_each_axis: If True, a scale factor will be chosen independently for each axis.
    """

    def __init__(self, *args, cl_key='cl', **kwargs):
        super().__init__(*args, **kwargs)
        self.cl_key = cl_key

    def __call__(self, **data_dict):
        data = data_dict.get(self.data_key)
        seg = data_dict.get(self.label_key)
        cl = data_dict.get(self.cl_key)
        # print(f"From spatial, data: {type(data)}, seg: {type(seg)}, cl: {type(cl)}") # all ndarray

        if self.patch_size is None:
            if len(data.shape) == 4:
                patch_size = (data.shape[2], data.shape[3])
            elif len(data.shape) == 5:
                patch_size = (data.shape[2], data.shape[3], data.shape[4])
            else:
                raise ValueError("only support 2D/3D batch data.")
        else:
            patch_size = self.patch_size

        # ret_val = augment_spatial(data, seg, patch_size=patch_size,
        #                           patch_center_dist_from_border=self.patch_center_dist_from_border,
        #                           do_elastic_deform=self.do_elastic_deform, alpha=self.alpha, sigma=self.sigma,
        #                           do_rotation=self.do_rotation, angle_x=self.angle_x, angle_y=self.angle_y,
        #                           angle_z=self.angle_z, do_scale=self.do_scale, scale=self.scale,
        #                           border_mode_data=self.border_mode_data,
        #                           border_cval_data=self.border_cval_data, order_data=self.order_data,
        #                           border_mode_seg=self.border_mode_seg, border_cval_seg=self.border_cval_seg,
        #                           order_seg=self.order_seg, random_crop=self.random_crop,
        #                           p_el_per_sample=self.p_el_per_sample, p_scale_per_sample=self.p_scale_per_sample,
        #                           p_rot_per_sample=self.p_rot_per_sample,
        #                           independent_scale_for_each_axis=self.independent_scale_for_each_axis,
        #                           p_rot_per_axis=self.p_rot_per_axis, 
        #                           p_independent_scale_per_axis=self.p_independent_scale_per_axis)
            ret_val = SpatialTransformCL.augment_spatial_cl(data, seg, cl, patch_size=patch_size,
                                                            patch_center_dist_from_border=self.patch_center_dist_from_border,
                                                            do_elastic_deform=self.do_elastic_deform, alpha=self.alpha, sigma=self.sigma,
                                                            do_rotation=self.do_rotation, angle_x=self.angle_x, angle_y=self.angle_y,
                                                            angle_z=self.angle_z, do_scale=self.do_scale, scale=self.scale,
                                                            border_mode_data=self.border_mode_data,
                                                            border_cval_data=self.border_cval_data, order_data=self.order_data,
                                                            border_mode_seg=self.border_mode_seg, border_cval_seg=self.border_cval_seg,
                                                            order_seg=self.order_seg, random_crop=self.random_crop,
                                                            p_el_per_sample=self.p_el_per_sample, p_scale_per_sample=self.p_scale_per_sample,
                                                            p_rot_per_sample=self.p_rot_per_sample,
                                                            independent_scale_for_each_axis=self.independent_scale_for_each_axis,
                                                            p_rot_per_axis=self.p_rot_per_axis, 
                                                            p_independent_scale_per_axis=self.p_independent_scale_per_axis)
            data_dict[self.data_key] = ret_val[0]
        if seg is not None:
            data_dict[self.label_key] = ret_val[1]
        if cl is not None:
            data_dict[self.cl_key] = ret_val[2]

        return data_dict
    
    @staticmethod
    def augment_spatial_cl(data, seg, cl, patch_size, patch_center_dist_from_border=30,
                    do_elastic_deform=True, alpha=(0., 1000.), sigma=(10., 13.),
                    do_rotation=True, angle_x=(0, 2 * np.pi), angle_y=(0, 2 * np.pi), angle_z=(0, 2 * np.pi),
                    do_scale=True, scale=(0.75, 1.25), border_mode_data='nearest', border_cval_data=0, order_data=3,
                    border_mode_seg='constant', border_cval_seg=0, order_seg=0, random_crop=True, p_el_per_sample=1,
                    p_scale_per_sample=1, p_rot_per_sample=1, independent_scale_for_each_axis=False,
                    p_rot_per_axis: float = 1, p_independent_scale_per_axis: int = 1):
        dim = len(patch_size)
        seg_result = None
        if seg is not None:
            if dim == 2:
                seg_result = np.zeros((seg.shape[0], seg.shape[1], patch_size[0], patch_size[1]), dtype=np.float32)
            else:
                seg_result = np.zeros((seg.shape[0], seg.shape[1], patch_size[0], patch_size[1], patch_size[2]),
                                    dtype=np.float32)
        if cl is not None:
            if dim == 2:
                cl_result = np.zeros((cl.shape[0], cl.shape[1], patch_size[0], patch_size[1]), dtype=np.float32)
            else:
                cl_result = np.zeros((cl.shape[0], cl.shape[1], patch_size[0], patch_size[1], patch_size[2]),
                                    dtype=np.float32)

        if dim == 2:
            data_result = np.zeros((data.shape[0], data.shape[1], patch_size[0], patch_size[1]), dtype=np.float32)
        else:
            data_result = np.zeros((data.shape[0], data.shape[1], patch_size[0], patch_size[1], patch_size[2]),
                                dtype=np.float32)

        if not isinstance(patch_center_dist_from_border, (list, tuple, np.ndarray)):
            patch_center_dist_from_border = dim * [patch_center_dist_from_border]

        for sample_id in range(data.shape[0]):
            coords = create_zero_centered_coordinate_mesh(patch_size)
            modified_coords = False

            if do_elastic_deform and np.random.uniform() < p_el_per_sample:
                a = np.random.uniform(alpha[0], alpha[1])
                s = np.random.uniform(sigma[0], sigma[1])
                coords = elastic_deform_coordinates(coords, a, s)
                modified_coords = True

            if do_rotation and np.random.uniform() < p_rot_per_sample:

                if np.random.uniform() <= p_rot_per_axis:
                    a_x = np.random.uniform(angle_x[0], angle_x[1])
                else:
                    a_x = 0

                if dim == 3:
                    if np.random.uniform() <= p_rot_per_axis:
                        a_y = np.random.uniform(angle_y[0], angle_y[1])
                    else:
                        a_y = 0

                    if np.random.uniform() <= p_rot_per_axis:
                        a_z = np.random.uniform(angle_z[0], angle_z[1])
                    else:
                        a_z = 0

                    coords = rotate_coords_3d(coords, a_x, a_y, a_z)
                else:
                    coords = rotate_coords_2d(coords, a_x)
                modified_coords = True

            if do_scale and np.random.uniform() < p_scale_per_sample:
                if independent_scale_for_each_axis and np.random.uniform() < p_independent_scale_per_axis:
                    sc = []
                    for _ in range(dim):
                        if np.random.random() < 0.5 and scale[0] < 1:
                            sc.append(np.random.uniform(scale[0], 1))
                        else:
                            sc.append(np.random.uniform(max(scale[0], 1), scale[1]))
                else:
                    if np.random.random() < 0.5 and scale[0] < 1:
                        sc = np.random.uniform(scale[0], 1)
                    else:
                        sc = np.random.uniform(max(scale[0], 1), scale[1])

                coords = scale_coords(coords, sc)
                modified_coords = True

            # now find a nice center location 
            if modified_coords:
                for d in range(dim):
                    if random_crop:
                        ctr = np.random.uniform(patch_center_dist_from_border[d],
                                                data.shape[d + 2] - patch_center_dist_from_border[d])
                    else:
                        ctr = data.shape[d + 2] / 2. - 0.5
                    coords[d] += ctr
                for channel_id in range(data.shape[1]):
                    data_result[sample_id, channel_id] = interpolate_img(data[sample_id, channel_id], coords, order_data,
                                                                        border_mode_data, cval=border_cval_data)
                if seg is not None:
                    for channel_id in range(seg.shape[1]):
                        seg_result[sample_id, channel_id] = interpolate_img(seg[sample_id, channel_id], coords, order_seg,
                                                                            border_mode_seg, cval=border_cval_seg,
                                                                            is_seg=True)
                if cl is not None:
                    for channel_id in range(cl.shape[1]):
                        cl_result[sample_id, channel_id] = interpolate_img(cl[sample_id, channel_id], coords, order_seg,
                                                                            mode=border_mode_seg, cval=border_cval_seg,
                                                                            is_seg=True)
                        
            else:
                if seg is None:
                    s = None
                    c = None
                else:
                    s = seg[sample_id:sample_id + 1]
                    c = cl[sample_id:sample_id + 1]
                if random_crop:
                    margin = [patch_center_dist_from_border[d] - patch_size[d] // 2 for d in range(dim)]
                    d, s, c = SpatialTransformCL.random_crop_cl(data[sample_id:sample_id + 1], s, c, patch_size, margin)
                else:
                    d, s, c = SpatialTransformCL.center_crop_cl(data[sample_id:sample_id + 1], patch_size, s, c)
                data_result[sample_id] = d[0]
                if seg is not None:
                    seg_result[sample_id] = s[0]
                if cl is not None:
                    cl_result[sample_id] = c[0]

        return data_result, seg_result, cl_result
    
    @staticmethod
    def center_crop_cl(data, crop_size, seg=None, cl=None):
        return SpatialTransformCL.crop_cl(data, seg, cl, crop_size, 0, 'center')
    
    @staticmethod
    def random_crop_cl(data, seg=None, cl=None, crop_size=128, margins=[0, 0, 0]):
        return SpatialTransformCL.crop_cl(data, seg, cl, crop_size, margins, 'random')
    
    @staticmethod
    def crop_cl(data, seg=None, cl=None, crop_size=128, margins=(0, 0, 0), crop_type="center",
         pad_mode='constant', pad_kwargs={'constant_values': 0},
         pad_mode_seg='constant', pad_kwargs_seg={'constant_values': 0},
         pad_mode_cl='constant', pad_kwargs_cl={'constant_values': 0}):
        """`
        crops data and seg (seg may be None) to crop_size. Whether this will be achieved via center or random crop is
        determined by crop_type. Margin will be respected only for random_crop and will prevent the crops form being closer
        than margin to the respective image border. crop_size can be larger than data_shape - margin -> data/seg will be
        padded with zeros in that case. margins can be negative -> results in padding of data/seg followed by cropping with
        margin=0 for the appropriate axes

        :param data: b, c, x, y(, z)
        :param seg:
        :param crop_size:
        :param margins: distance from each border, can be int or list/tuple of ints (one element for each dimension).
        Can be negative (data/seg will be padded if needed)
        :param crop_type: random or center
        :return:
        """
        if not isinstance(data, (list, tuple, np.ndarray)):
            raise TypeError("data has to be either a numpy array or a list")

        data_shape = tuple([len(data)] + list(data[0].shape))
        data_dtype = data[0].dtype
        dim = len(data_shape) - 2

        if seg is not None:
            seg_shape = tuple([len(seg)] + list(seg[0].shape))
            seg_dtype = seg[0].dtype

            if not isinstance(seg, (list, tuple, np.ndarray)):
                raise TypeError("data has to be either a numpy array or a list")

            assert all([i == j for i, j in zip(seg_shape[2:], data_shape[2:])]), "data and seg must have the same spatial " \
                                                                                "dimensions. Data: %s, seg: %s" % \
                                                                                (str(data_shape), str(seg_shape))
        
        if cl is not None:
            cl_shape = tuple([len(cl)] + list(cl[0].shape))
            cl_dtype = cl[0].dtype
            
            if not isinstance(seg, (list, tuple, np.ndarray)):
                raise TypeError("data has to be either a numpy array or a list")

            assert all([i == j for i, j in zip(seg_shape[2:], cl_shape[2:])]), "seg and cl must have the same spatial " \
                                                                                "dimensions. Seg: %s, Data: %s" % \
                                                                                (str(seg_shape), str(cl_shape))
            
        # print(f"before crop, data: {data_shape}, seg: {seg_shape}, cl: {cl_shape}")

        if type(crop_size) not in (tuple, list, np.ndarray):
            crop_size = [crop_size] * dim
        else:
            assert len(crop_size) == len(
                data_shape) - 2, "If you provide a list/tuple as center crop make sure it has the same dimension as your " \
                                "data (2d/3d)"

        if not isinstance(margins, (np.ndarray, tuple, list)):
            margins = [margins] * dim

        data_return = np.zeros([data_shape[0], data_shape[1]] + list(crop_size), dtype=data_dtype)
        if seg is not None:
            seg_return = np.zeros([seg_shape[0], seg_shape[1]] + list(crop_size), dtype=seg_dtype)
        else:
            seg_return = None
        if cl is not None:
            cl_return = np.zeros([cl_shape[0], cl_shape[1]] + list(crop_size), dtype=cl_dtype)
        else:
            cl_return = None

        for b in range(data_shape[0]):
            data_shape_here = [data_shape[0]] + list(data[b].shape)
            if seg is not None:
                seg_shape_here = [seg_shape[0]] + list(seg[b].shape)
            if cl is not None:
                cl_shape_here = [cl_shape[0]] + list(cl[b].shape)

            if crop_type == "center":
                lbs = get_lbs_for_center_crop(crop_size, data_shape_here)
            elif crop_type == "random":
                lbs = get_lbs_for_random_crop(crop_size, data_shape_here, margins)
            else:
                raise NotImplementedError("crop_type must be either center or random")

            need_to_pad = [[0, 0]] + [[abs(min(0, lbs[d])),
                                    abs(min(0, data_shape_here[d + 2] - (lbs[d] + crop_size[d])))]
                                    for d in range(dim)]

            # we should crop first, then pad -> reduces i/o for memmaps, reduces RAM usage and improves speed
            ubs = [min(lbs[d] + crop_size[d], data_shape_here[d+2]) for d in range(dim)]
            lbs = [max(0, lbs[d]) for d in range(dim)]

            slicer_data = [slice(0, data_shape_here[1])] + [slice(lbs[d], ubs[d]) for d in range(dim)]
            data_cropped = data[b][tuple(slicer_data)]

            if seg_return is not None:
                slicer_seg = [slice(0, seg_shape_here[1])] + [slice(lbs[d], ubs[d]) for d in range(dim)]
                seg_cropped = seg[b][tuple(slicer_seg)]

            if cl_return is not None:
                slicer_cl = [slice(0, cl_shape_here[2])] + [slice(lbs[d], ubs[d]) for d in range(dim)]
                cl_cropped = cl[b][tuple(slicer_cl)]

            if any([i > 0 for j in need_to_pad for i in j]):
                data_return[b] = np.pad(data_cropped, need_to_pad, pad_mode, **pad_kwargs)
                if seg_return is not None:
                    seg_return[b] = np.pad(seg_cropped, need_to_pad, pad_mode_seg, **pad_kwargs_seg)
                if cl_return is not None:
                    cl_return[b] = np.pad(cl_cropped, need_to_pad, pad_mode_cl, **pad_kwargs_cl)
            else:
                data_return[b] = data_cropped
                if seg_return is not None:
                    seg_return[b] = seg_cropped
                if cl_return is not None:
                    cl_return[b] = cl_cropped
        
        # print(f"from crop_cl, data: {data_return.shape}, seg: {seg_return.shape}, cl: {cl_return.shape}") # all same size!
        return data_return, seg_return, cl_return
