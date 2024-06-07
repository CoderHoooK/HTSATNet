import numpy as np
import copy as cp
from torchvision.transforms import Compose
import math
"""
这个加上fft
"""
class PreNormalize3D:
    """PreNormalize for NTURGB+D 3D keypoints (x, y, z). Codes adapted from https://github.com/lshiwjx/2s-AGCN. """

    def unit_vector(self, vector):
        """Returns the unit vector of the vector. """
        return vector / np.linalg.norm(vector)

    def angle_between(self, v1, v2):
        """Returns the angle in radians between vectors 'v1' and 'v2'. """
        if np.abs(v1).sum() < 1e-6 or np.abs(v2).sum() < 1e-6:
            return 0
        v1_u = self.unit_vector(v1)
        v2_u = self.unit_vector(v2)
        return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

    def rotation_matrix(self, axis, theta):
        """Return the rotation matrix associated with counterclockwise rotation
        about the given axis by theta radians."""
        if np.abs(axis).sum() < 1e-6 or np.abs(theta) < 1e-6:
            return np.eye(3)
        axis = np.asarray(axis)
        axis = axis / np.sqrt(np.dot(axis, axis))
        a = np.cos(theta / 2.0)
        b, c, d = -axis * np.sin(theta / 2.0)
        aa, bb, cc, dd = a * a, b * b, c * c, d * d
        bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
        return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                        [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                        [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

    def __init__(self, zaxis=[0, 1], xaxis=[8, 4], align_spine=True, align_center=True):
        
        self.zaxis = zaxis
        self.xaxis = xaxis
        self.align_spine = align_spine
        self.align_center = align_center

    def __call__(self, results):
        skeleton = results
        total_frames = skeleton.shape[0]
        T, V, C = skeleton.shape
        if self.align_center:
            #1是掌心
            main_body_center = skeleton[0, 1].copy()
            mask = ((skeleton != 0).sum(-1) > 0)[..., None]
            skeleton = (skeleton - main_body_center) * mask

        if self.align_spine:
            joint_bottom = skeleton[0, 0, self.zaxis[0]]
            joint_top = skeleton[0, 0, self.zaxis[1]]
            axis = np.cross(joint_top - joint_bottom, [0, 0, 1])
            angle = self.angle_between(joint_top - joint_bottom, [0, 0, 1])
            matrix_z = self.rotation_matrix(axis, angle)
            skeleton = np.einsum('abcd,kd->abck', skeleton, matrix_z)

            joint_rshoulder = skeleton[0, 0, self.xaxis[0]]
            joint_lshoulder = skeleton[0, 0, self.xaxis[1]]
            axis = np.cross(joint_rshoulder - joint_lshoulder, [1, 0, 0])
            angle = self.angle_between(joint_rshoulder - joint_lshoulder, [1, 0, 0])
            matrix_x = self.rotation_matrix(axis, angle)
            skeleton = np.einsum('abcd,kd->abck', skeleton, matrix_x)

        return results

class RandomRot:

    def __init__(self, theta=0.3):
        self.theta = theta

    def _rot3d(self, theta):
        cos, sin = np.cos(theta), np.sin(theta)
        rx = np.array([[1, 0, 0], [0, cos[0], sin[0]], [0, -sin[0], cos[0]]])
        ry = np.array([[cos[1], 0, -sin[1]], [0, 1, 0], [sin[1], 0, cos[1]]])
        rz = np.array([[cos[2], sin[2], 0], [-sin[2], cos[2], 0], [0, 0, 1]])

        rot = np.matmul(rz, np.matmul(ry, rx))
        return rot

    def _rot2d(self, theta):
        cos, sin = np.cos(theta), np.sin(theta)
        return np.array([[cos, -sin], [sin, cos]])

    def __call__(self, results):
        skeleton = results
        T, V, C = skeleton.shape
        theta = np.random.uniform(-self.theta, self.theta, size=3)
        rot_mat = self._rot3d(theta)
        results = np.einsum('ab,tvb->tva', rot_mat, skeleton)
        return results
    

class JointToBone:

    def __init__(self, dataset='dhg'):
        self.dataset = dataset
        if self.dataset == 'dhg':
            self.pairs = [(1, 2), (3, 1), (4, 3), (5, 4), (6, 5), (7, 2), (8, 7), (9, 8), (10, 9), (11, 2), (12, 11),
                     (13, 12), (14, 13), (15, 2), (16, 15), (17, 16), (18, 17), (19, 2), (20, 19), (21, 20), (22, 21),
                     (2, 2)]

    def __call__(self, results):

        keypoint = results
        T, V, C = keypoint.shape
        bone = np.zeros((T, V, C), dtype=np.float32)

        assert C in [2, 3]
        for v1, v2 in self.pairs:
            v1 = v1 -1
            v2 = v2 -1
            bone[..., v1, :] = keypoint[..., v1, :] - keypoint[..., v2, :]
        return bone

class ToMotion:

    def __init__(self, dataset='dhg'):
        self.dataset = dataset

    def __call__(self, results):
        data = results
        T, V, C = data.shape
        motion = np.zeros_like(data)

        assert C in [2, 3]
        motion[:T - 1] = np.diff(data, axis=0)

        return motion
    

class Rename:
    """Rename the key in results.

    Args:
        mapping (dict): The keys in results that need to be renamed. The key of
            the dict is the original name, while the value is the new name. If
            the original name not found in results, do nothing.
            Default: dict().
    """

    def __init__(self, mapping):
        self.mapping = mapping

    def __call__(self, results):
        for key, value in self.mapping.items():
            if key in results:
                assert isinstance(key, str) and isinstance(value, str)
                assert value not in results, ('the new name already exists in '
                                              'results')
                results[value] = results[key]
                results.pop(key)
        return results

class MergeSkeFeat:
    def __init__(self, feat_list=['keypoint'], target='keypoint', axis=-1):
        """Merge different feats (ndarray) by concatenate them in the last axis. """

        self.feat_list = feat_list
        self.target = target
        self.axis = axis

    def __call__(self, results):
        feats = []
        for name in self.feat_list:
            feats.append(results.pop(name))
        feats = np.concatenate(feats, axis=self.axis)
        results[self.target] = feats
        return results


class ToImag:

    def __init__(self, dataset='dhg', source='keypoint', target='imag'):
        self.dataset = dataset
        self.source = source
        self.target = target

    def __call__(self, results):
        data = results
        fft = np.fft.fft(data,axis=1)

        results = fft.imag

        return results
    
class ToReal:

    def __init__(self, dataset='dhg', source='keypoint', target='real'):
        self.dataset = dataset
        self.source = source
        self.target = target

    def __call__(self, results):
        data = results
        fft = np.fft.fft(data,axis=1)
        results = fft.real

        return results
    
class ToAP:

    def __init__(self, dataset='dhg', source='keypoint', target='ja'):
        self.dataset = dataset
        self.source = source
        self.target = target

    def __call__(self, results):
        data = results
        fft = np.fft.fft(data,axis=1)
        results = np.abs(fft)

        return results
class GenSkeFeat:
    #增加了joint和bone的 imag real 和ri特征
    def __init__(self, dataset='dhg', feats=['j'], axis=-1):
        self.dataset = dataset
        self.feats = feats
        self.axis = axis
        ops = []
        if 'b' in feats or 'bm' in feats:
            ops.append(JointToBone(dataset=dataset))
        if 'jm' in feats:
            ops.append(ToMotion(dataset=dataset))
        if 'bm' in feats:
            ops.append(ToMotion(dataset=dataset))
        # ops.append(MergeSkeFeat(feat_list=feats, axis=axis))
        if 'ji' in feats:
            ops.append(ToImag(dataset=dataset, source='j', target='ji'))
        if 'jr' in feats:
            ops.append(ToReal(dataset=dataset, source='j', target='jr'))   
        if 'ja' in feats:
            ops.append(ToAP(dataset=dataset, source='j', target='ja'))      
        self.ops = Compose(ops)

    def __call__(self, results):
        return self.ops(results)

class UniformSampleDecode:
    def __init__(self, clip_len,seed = 255, num_clips=1, p_interval=1,test_mode = False):
        self.clip_len = clip_len
        self.num_clips = num_clips
        self.seed = seed
        self.test_mode = test_mode
        self.p_interval = p_interval
        if not isinstance(p_interval, list):
            self.p_interval = (p_interval, p_interval)
    # will directly return the decoded clips
    def _get_clips(self, full_kp, clip_len):
        T, V, C = full_kp.shape
        clips = []
    
        for clip_idx in range(self.num_clips):
            pi = self.p_interval
            ratio = np.random.rand() * (pi[1] - pi[0]) + pi[0]
            num_frames = int(ratio * T)
            off = np.random.randint(T - num_frames + 1)

            if num_frames < clip_len:
                start = np.random.randint(0, num_frames)
                inds = (np.arange(start, start + clip_len) % num_frames) + off
                clip = full_kp[inds].copy()
            elif clip_len <= num_frames < 2 * clip_len:
                basic = np.arange(clip_len)
                inds = np.random.choice(clip_len + 1, num_frames - clip_len, replace=False)
                offset = np.zeros(clip_len + 1, dtype=np.int64)
                offset[inds] = 1
                inds = basic + np.cumsum(offset)[:-1] + off
                clip = full_kp[inds].copy()
            else:
                bsize = num_frames // clip_len
                split_num = num_frames//bsize
                if split_num > clip_len:
                    f1 = full_kp[off:split_num*bsize+off].reshape(bsize,split_num,V,C)
                    f1_mean = f1.mean(0)
                    if num_frames%bsize != 0: 
                        f2 = full_kp[split_num*bsize+off:].reshape(-1,1,V,C)
                        f2_mean = f2.mean(0)
                        f = np.concatenate((f1_mean,f2_mean),0)
                    else: f = f1_mean
                    mask = np.zeros(f.shape[0], dtype=bool)
                    select = np.random.choice(f.shape[0], clip_len , replace=False)
                    mask[select] = True
                    clip = f[mask].copy()
                else:
                    f1 = full_kp[off:split_num*bsize+off].reshape(bsize,split_num,V,C)
                    f1_mean = f1.mean(0)
                    f = f1_mean
                    clip = f
            clips.append(clip)
        return np.concatenate(clips, 0)

    def _handle(self, results):
        kp = results
        kp = kp.astype(np.float32)
        # start_index will not be used
        kp = self._get_clips(kp, self.clip_len)
        return kp


    def __call__(self, results):
        if self.test_mode is True:
            np.random.seed(self.seed)
        return self._handle(results)


class FormatGCNInput:
    """Format final skeleton shape to the given input_format. """

    def __init__(self, num_person=2, mode='zero'):
        self.num_person = num_person
        assert mode in ['zero', 'loop']
        self.mode = mode

    def __call__(self, results):
        """Performs the FormatShape formatting.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        keypoint = results

        # T V C
        T, V, C = keypoint.shape
        nc = results.get('num_clips', 1)
        assert T % nc == 0
        keypoint = keypoint.reshape((M, nc, T // nc, V, C)).transpose(1, 0, 2, 3, 4)
        results['keypoint'] = np.ascontiguousarray(keypoint)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__ + f'(num_person={self.num_person}, mode={self.mode})'
        return repr_str

class Collect:
    """Collect data from the loader relevant to the specific task.

    This keeps the items in ``keys`` as it is, and collect items in
    ``meta_keys`` into a meta item called ``meta_name``.This is usually
    the last stage of the data loader pipeline.
    For example, when keys='imgs', meta_keys=('filename', 'label',
    'original_shape'), meta_name='img_metas', the results will be a dict with
    keys 'imgs' and 'img_metas', where 'img_metas' is a DataContainer of
    another dict with keys 'filename', 'label', 'original_shape'.

    Args:
        keys (Sequence[str]): Required keys to be collected.
        meta_name (str): The name of the key that contains meta information.
            This key is always populated. Default: "img_metas".
        meta_keys (Sequence[str]): Keys that are collected under meta_name.
            The contents of the ``meta_name`` dictionary depends on
            ``meta_keys``.
            By default this includes:

            - "filename": path to the image file
            - "label": label of the image file
            - "original_shape": original shape of the image as a tuple
                (h, w, c)
            - "img_shape": shape of the image input to the network as a tuple
                (h, w, c).  Note that images may be zero padded on the
                bottom/right, if the batch tensor is larger than this shape.
            - "pad_shape": image shape after padding
            - "flip_direction": a str in ("horiziontal", "vertival") to
                indicate if the image is fliped horizontally or vertically.
            - "img_norm_cfg": a dict of normalization information:
                - mean - per channel mean subtraction
                - std - per channel std divisor
                - to_rgb - bool indicating if bgr was converted to rgb
        nested (bool): If set as True, will apply data[x] = [data[x]] to all
            items in data. The arg is added for compatibility. Default: False.
    """

    def __init__(self,
                 keys,
                 meta_keys=('filename', 'label', 'original_shape', 'img_shape',
                            'pad_shape', 'flip_direction', 'img_norm_cfg'),
                 meta_name='img_metas',
                 nested=False):
        self.keys = keys
        self.meta_keys = meta_keys
        self.meta_name = meta_name
        self.nested = nested

    def __call__(self, results):
        """Performs the Collect formatting.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        data = {}
        for key in self.keys:
            data[key] = results[key]

        if len(self.meta_keys) != 0:
            meta = {}
            for key in self.meta_keys:
                meta[key] = results[key]
            data[self.meta_name] = DC(meta, cpu_only=True)
        if self.nested:
            for k in data:
                data[k] = [data[k]]

        return data

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'keys={self.keys}, meta_keys={self.meta_keys}, '
                f'nested={self.nested})')


