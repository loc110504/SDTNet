# dataloader/acdc_xnet.py
import itertools
import os
import random
import re
from glob import glob
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
import pywt

from random import sample


def pseudo_label_generator_acdc(data, seed, beta=100, mode='bf'):
    from skimage.exposure import rescale_intensity
    from skimage.segmentation import random_walker
    if 1 not in np.unique(seed) or 2 not in np.unique(seed) or 3 not in np.unique(seed):
        pseudo_label = np.zeros_like(seed)
    else:
        markers = np.ones_like(seed)
        markers[seed == 4] = 0
        markers[seed == 0] = 1
        markers[seed == 1] = 2
        markers[seed == 2] = 3
        markers[seed == 3] = 4
        sigma = 0.35
        data = rescale_intensity(data, in_range=(-sigma, 1 + sigma),
                                 out_range=(-1, 1))
        segmentation = random_walker(data, markers, beta, mode)
        pseudo_label = segmentation - 1
    return pseudo_label


# =========================
#  High–Low utilities
# =========================
def _safe_minmax(x, axes=(0, 1), eps=1e-8, out_scale=1.0):
    """
    Chuẩn hóa min-max theo các trục không gian, an toàn chia 0.
    Trả về cùng shape như x, trong [0, out_scale].
    """
    x_min = np.amin(x, axis=axes, keepdims=True)
    x_max = np.amax(x, axis=axes, keepdims=True)
    denom = np.maximum(x_max - x_min, eps)
    x_n = (x - x_min) / denom
    return x_n * out_scale


def compute_hilo_from_image2d(img2d,
                              wavelet_type='haar',
                              alpha_range=(0.1, 0.3),
                              beta_range=(0.05, 0.2),
                              rng=None):
    """
    Tạo đặc trưng L/H từ ảnh xám 2D bằng DWT2.
    - Chuẩn hóa L/H về [0,1] (float32).
    - Resize L/H về đúng kích thước ảnh gốc (do DWT giảm nửa kích thước).
    """
    assert img2d.ndim in (2, 3)
    if img2d.ndim == 3 and img2d.shape[-1] == 1:
        img2d = img2d[..., 0]

    # DWT2 trên không gian (H,W) -> hệ số có kích thước ~ H/2 x W/2
    LL, (LH, HL, HH) = pywt.dwt2(img2d, wavelet_type, axes=(0, 1))

    LL = _safe_minmax(LL, (0, 1), out_scale=1.0)
    LH = _safe_minmax(LH, (0, 1), out_scale=1.0)
    HL = _safe_minmax(HL, (0, 1), out_scale=1.0)
    HH = _safe_minmax(HH, (0, 1), out_scale=1.0)

    H_ = HL + LH + HH
    H_ = _safe_minmax(H_, (0, 1), out_scale=1.0)

    rng = np.random.default_rng() if rng is None else rng
    L_alpha = rng.uniform(alpha_range[0], alpha_range[1])
    H_beta = rng.uniform(beta_range[0], beta_range[1])

    L = LL + L_alpha * H_
    L = _safe_minmax(L, (0, 1), out_scale=1.0)

    H = H_ + H_beta * LL
    H = _safe_minmax(H, (0, 1), out_scale=1.0)

    # Resize L/H về đúng shape ảnh gốc
    if L.shape != img2d.shape:
        sx = img2d.shape[0] / L.shape[0]
        sy = img2d.shape[1] / L.shape[1]
        # Dùng nội suy bậc 1 cho đặc trưng liên tục
        L = zoom(L, (sx, sy), order=1)
        H = zoom(H, (sx, sy), order=1)

    return L.astype(np.float32), H.astype(np.float32)


# =========================
#  ACDC Dataset Loader
# =========================
class BaseDataSets(Dataset):
    """
    Base class for ACDC dataset loader. Choose the suitable fold and split
    """
    def __init__(self, base_dir=None, split='train', transform=None,
                 fold="fold1", sup_type="label",
                 enable_hilo=False,
                 wavelet_type='haar',
                 alpha_range=(0.1, 0.3),
                 beta_range=(0.05, 0.2),
                 rng=None):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.sup_type = sup_type
        self.transform = transform

        # cấu hình High–Low
        self.enable_hilo = enable_hilo
        self.wavelet_type = wavelet_type
        self.alpha_range = alpha_range
        self.beta_range = beta_range
        self.rng = rng

        train_ids, test_ids = self._get_fold_ids(fold)
        if self.split == 'train':
            self.all_slices = os.listdir(
                self._base_dir + "/ACDC_training_slices")
            self.sample_list = []
            for ids in train_ids:
                new_data_list = list(filter(lambda x: re.match(
                    '{}.*'.format(ids), x) is not None, self.all_slices))
                self.sample_list.extend(new_data_list)

        elif self.split == 'val':
            self.all_volumes = os.listdir(
                self._base_dir + "/ACDC_training_volumes")
            self.sample_list = []
            for ids in test_ids:
                new_data_list = list(filter(lambda x: re.match(
                    '{}.*'.format(ids), x) is not None, self.all_volumes))
                self.sample_list.extend(new_data_list)

        print("total {} samples".format(len(self.sample_list)))

    def _get_fold_ids(self, fold):
        """
        Fold1:
            - Training: patient021 to patient100
            - Testing: patient001 to patient020
        """
        all_cases_set = ["patient{:0>3}".format(i) for i in range(1, 101)]
        fold1_testing_set = [
            "patient{:0>3}".format(i) for i in range(1, 21)]
        fold1_training_set = [
            i for i in all_cases_set if i not in fold1_testing_set]
        fold2_testing_set = [
            "patient{:0>3}".format(i) for i in range(21, 41)]
        fold2_training_set = [
            i for i in all_cases_set if i not in fold2_testing_set]

        fold3_testing_set = [
            "patient{:0>3}".format(i) for i in range(41, 61)]
        fold3_training_set = [
            i for i in all_cases_set if i not in fold3_testing_set]

        fold4_testing_set = [
            "patient{:0>3}".format(i) for i in range(61, 81)]
        fold4_training_set = [
            i for i in all_cases_set if i not in fold4_testing_set]

        fold5_testing_set = [
            "patient{:0>3}".format(i) for i in range(81, 101)]
        fold5_training_set = [
            i for i in all_cases_set if i not in fold5_testing_set]
        if fold == "fold1":
            return [fold1_training_set, fold1_testing_set]
        elif fold == "fold2":
            return [fold2_training_set, fold2_testing_set]
        elif fold == "fold3":
            return [fold3_training_set, fold3_testing_set]
        elif fold == "fold4":
            return [fold4_training_set, fold4_testing_set]
        elif fold == "fold5":
            return [fold5_training_set, fold5_testing_set]
        elif fold == "MAAGfold":
            training_set = ["patient{:0>3}".format(i) for i in [37, 50, 53, 100, 38, 19, 61, 74, 97, 31, 91, 35, 56, 94, 26, 69, 46, 59, 4, 89,
                                71, 6, 52, 43, 45, 63, 93, 14, 98, 88, 21, 28, 99, 54, 90]]
            validation_set = ["patient{:0>3}".format(i) for i in [84, 32, 27, 96, 17, 18, 57, 81, 79, 22, 1, 44, 49, 25, 95]]
            return [training_set, validation_set]
        elif fold == "MAAGfold70":
            training_set = ["patient{:0>3}".format(i) for i in [37, 50, 53, 100, 38, 19, 61, 74, 97, 31, 91, 35, 56, 94, 26, 69, 46, 59, 4, 89,
                                71, 6, 52, 43, 45, 63, 93, 14, 98, 88, 21, 28, 99, 54, 90, 2, 76, 34, 85, 70, 86, 3, 8, 51, 40, 7, 13, 47, 55, 12, 58, 87, 9, 65, 62, 33, 42,
                               23, 92, 29, 11, 83, 68, 75, 67, 16, 48, 66, 20, 15]]
            validation_set = ["patient{:0>3}".format(i) for i in [84, 32, 27, 96, 17, 18, 57, 81, 79, 22, 1, 44, 49, 25, 95]]
            return [training_set, validation_set]
        elif "MAAGfold" in fold:
            training_num = int(fold[8:])
            training_set = sample(["patient{:0>3}".format(i) for i in [37, 50, 53, 100, 38, 19, 61, 74, 97, 31, 91, 35, 56, 94, 26, 69, 46, 59, 4, 89,
                                71, 6, 52, 43, 45, 63, 93, 14, 98, 88, 21, 28, 99, 54, 90, 2, 76, 34, 85, 70, 86, 3, 8, 51, 40, 7, 13, 47, 55, 12, 58, 87, 9, 65, 62, 33, 42,
                               23, 92, 29, 11, 83, 68, 75, 67, 16, 48, 66, 20, 15]], training_num)
            print("total {} training samples: {}".format(training_num, training_set))
            validation_set = ["patient{:0>3}".format(i) for i in [84, 32, 27, 96, 17, 18, 57, 81, 79, 22, 1, 44, 49, 25, 95]]
            return [training_set, validation_set]
        else:
            return "ERROR KEY"

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        if self.split == "train":
            h5f = h5py.File(self._base_dir +
                            "/ACDC_training_slices/{}".format(case), 'r')
        else:
            h5f = h5py.File(self._base_dir +
                            "/ACDC_training_volumes/{}".format(case), 'r')

        # đọc ảnh/nhãn
        image = h5f['image'][:]  # train: thường [D,H,W]; val: [D,H,W]
        if self.split == "train":
            if self.sup_type == "random_walker":
                label = pseudo_label_generator_acdc(image, h5f["scribble"][:])
            else:
                label = h5f[self.sup_type][:]
        else:
            label = h5f['label'][:]

        if self.split == "train":
            # ---- Chọn 1 lát 2D để phù hợp Conv2d ----
            if image.ndim == 3:
                d = image.shape[0]
                ind = np.random.randint(d)
                image2d = image[ind, ...]
                label2d = label[ind, ...]
            else:
                image2d = image
                label2d = label

            # tạo High–Low nếu bật
            if self.enable_hilo:
                L, H = compute_hilo_from_image2d(
                    image2d,
                    wavelet_type=self.wavelet_type,
                    alpha_range=self.alpha_range,
                    beta_range=self.beta_range,
                    rng=self.rng
                )
                sample = {'image': image2d, 'label': label2d, 'L': L, 'H': H}
            else:
                sample = {'image': image2d, 'label': label2d}

            # transform (augment → tensor)
            if self.transform is not None:
                sample = self.transform(sample)
            else:
                image_t = torch.from_numpy(image2d.astype(np.float32)).unsqueeze(0)
                label_t = torch.from_numpy(label2d.astype(np.uint8))
                sample = {'image': image_t, 'label': label_t}
                if self.enable_hilo:
                    sample['L'] = torch.from_numpy(L.astype(np.float32)).unsqueeze(0)
                    sample['H'] = torch.from_numpy(H.astype(np.float32)).unsqueeze(0)

        else:
            # ---- VAL: GIỮ 3D để hàm test xử lý theo lát ----
            # image: [D,H,W] -> [1,D,H,W]
            image_t = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
            label_t = torch.from_numpy(label.astype(np.uint8))
            sample = {'image': image_t, 'label': label_t}
            # (val không dùng High–Low)

        sample["idx"] = idx
        return sample


# =========================
#  Augment helpers
# =========================
def random_rot_flip_2d(arr, k=None, axis=None):
    """Deterministic rot90+flip cho 2D."""
    if k is None:
        k = np.random.randint(0, 4)
    if axis is None:
        axis = np.random.randint(0, 2)
    out = np.rot90(arr, k)
    out = np.flip(out, axis=axis).copy()
    return out, k, axis


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        """
        Đồng bộ phép biến đổi cho image/label và (nếu có) L/H.
        - rot90 + flip với cùng k/axis
        - rotate với cùng angle
        - zoom về output_size
        Kỳ vọng: image/label/L/H đều là 2D numpy trước khi vào đây.
        """
        image, label = sample['image'], sample['label']
        has_hilo = ('L' in sample) and ('H' in sample)
        if has_hilo:
            L, H = sample['L'], sample['H']

        # --- quyết định augment đồng bộ ---
        do_rf = (random.random() > 0.5)
        do_rot = (not do_rf) and (random.random() > 0.5)

        if do_rf:
            k = np.random.randint(0, 4)
            axis = np.random.randint(0, 2)
            image = np.rot90(image, k);        label = np.rot90(label, k)
            image = np.flip(image, axis=axis).copy()
            label = np.flip(label, axis=axis).copy()
            if has_hilo:
                L = np.rot90(L, k);            H = np.rot90(H, k)
                L = np.flip(L, axis=axis).copy()
                H = np.flip(H, axis=axis).copy()

        elif do_rot:
            angle = np.random.randint(-20, 20)
            cval = 4 if (4 in np.unique(label)) else 0
            image = ndimage.rotate(image, angle, order=0, reshape=False)
            label = ndimage.rotate(label, angle, order=0, reshape=False, mode="constant", cval=cval)
            if has_hilo:
                L = ndimage.rotate(L, angle, order=1, reshape=False)
                H = ndimage.rotate(H, angle, order=1, reshape=False)

        # --- resize/zoom đồng bộ ---
        x, y = image.shape
        zx, zy = (self.output_size[0] / x, self.output_size[1] / y)
        image = zoom(image, (zx, zy), order=0)
        label = zoom(label, (zx, zy), order=0)
        if has_hilo:
            L = zoom(L, (zx, zy), order=1)
            H = zoom(H, (zx, zy), order=1)

        # --- to tensor ---
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)  # [1,H,W]
        label = torch.from_numpy(label.astype(np.uint8))                 # [H,W]
        out = {'image': image, 'label': label}
        if has_hilo:
            out['L'] = torch.from_numpy(L.astype(np.float32)).unsqueeze(0)  # [1,H,W]
            out['H'] = torch.from_numpy(H.astype(np.float32)).unsqueeze(0)  # [1,H,W]
        return out


# =========================
#  Sampler (nguyên gốc)
# =========================
class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """

    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                   grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)
