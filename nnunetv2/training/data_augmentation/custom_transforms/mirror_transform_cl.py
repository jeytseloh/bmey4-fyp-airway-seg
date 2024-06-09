from batchgenerators.transforms.abstract_transforms import AbstractTransform
# from batchgenerators.augmentations.spatial_transformations import augment_mirroring
import numpy as np

class MirrorTransformCL(AbstractTransform):
    """ Randomly mirrors data along specified axes. Mirroring is evenly distributed. Probability of mirroring along
    each axis is 0.5

    Args:
        axes (tuple of int): axes along which to mirror

    """

    def __init__(self, axes=(0, 1, 2), data_key="data", label_key="seg", cl_key="cl", p_per_sample=1):
        self.p_per_sample = p_per_sample
        self.data_key = data_key
        self.label_key = label_key
        self.cl_key = cl_key
        self.axes = axes
        if max(axes) > 2:
            raise ValueError("MirrorTransform now takes the axes as the spatial dimensions. What previously was "
                             "axes=(2, 3, 4) to mirror along all spatial dimensions of a 5d tensor (b, c, x, y, z) "
                             "is now axes=(0, 1, 2). Please adapt your scripts accordingly.")

    def __call__(self, **data_dict):
        data = data_dict.get(self.data_key)
        seg = data_dict.get(self.label_key)
        cl = data_dict.get(self.cl_key)

        for b in range(len(data)):
            if np.random.uniform() < self.p_per_sample:
                sample_seg = None
                sample_cl = None
                if seg is not None:
                    sample_seg = seg[b]
                if cl is not None:
                    sample_cl = cl[b]
                ret_val = MirrorTransformCL.augment_mirroring_cl(data[b], sample_seg, sample_cl, axes=self.axes)
                data[b] = ret_val[0]
                if seg is not None:
                    seg[b] = ret_val[1]
                if cl is not None:
                    cl[b] = ret_val[2]

        data_dict[self.data_key] = data
        if seg is not None:
            data_dict[self.label_key] = seg
        if cl is not None:
            data_dict[self.cl_key] = cl

        return data_dict
    
    @staticmethod
    def augment_mirroring_cl(sample_data, sample_seg=None, sample_cl=None, axes=(0, 1, 2)):
        if (len(sample_data.shape) != 3) and (len(sample_data.shape) != 4):
            raise Exception(
                "Invalid dimension for sample_data and sample_seg. sample_data and sample_seg should be either "
                "[channels, x, y] or [channels, x, y, z]")
        if 0 in axes and np.random.uniform() < 0.5:
            sample_data[:, :] = sample_data[:, ::-1]
            if sample_seg is not None:
                sample_seg[:, :] = sample_seg[:, ::-1]
            if sample_cl is not None: # centreline
                sample_cl[:, :] = sample_cl[:, ::-1]
        if 1 in axes and np.random.uniform() < 0.5:
            sample_data[:, :, :] = sample_data[:, :, ::-1]
            if sample_seg is not None:
                sample_seg[:, :, :] = sample_seg[:, :, ::-1]
            if sample_cl is not None: # centreline
                sample_cl[:, :, :] = sample_cl[:, :, ::-1]
        if 2 in axes and len(sample_data.shape) == 4:
            if np.random.uniform() < 0.5:
                sample_data[:, :, :, :] = sample_data[:, :, :, ::-1]
                if sample_seg is not None:
                    sample_seg[:, :, :, :] = sample_seg[:, :, :, ::-1]
                if sample_cl is not None: # centreline
                    sample_cl[:, :, :, :] = sample_cl[:, :, :, ::-1]
        return sample_data, sample_seg, sample_cl