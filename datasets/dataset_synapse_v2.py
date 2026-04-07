import os
import random

import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset


from scipy.ndimage.interpolation import zoom

# 1. HÀM LẬT VÀ XOAY 90 ĐỘ (Hỗ trợ 3 Kênh)
def random_rot_flip(image, label):
    # image shape: (3, H, W) | label shape: (H, W)
    k = np.random.randint(0, 4)
    
    # Xoay mặt phẳng (H, W) - tức là trục 1 và trục 2 của image
    image = np.rot90(image, k, axes=(1, 2))
    label = np.rot90(label, k, axes=(0, 1)) # label chỉ có 2D nên vẫn là 0, 1
    
    axis = np.random.randint(0, 2)
    # Lật theo H hoặc W (cộng thêm 1 vì image bị vướng cái số kênh ở đầu)
    image = np.flip(image, axis=axis + 1).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label

# 2. HÀM XOAY GÓC NGHIÊNG BẤT KỲ (Hỗ trợ 3 Kênh)
def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    
    # Bắt buộc khai báo axes=(1, 2) để chỉ xoay ảnh dọc/ngang, không xoay kênh
    image = ndimage.rotate(image, angle, axes=(1, 2), order=0, reshape=False)
    label = ndimage.rotate(label, angle, axes=(0, 1), order=0, reshape=False)
    return image, label

# 3. DÂY CHUYỀN NHÀO NẶN (Đã sửa lỗi kích thước)
class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # Xác suất nhào nặn
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
            
        # Lấy kích thước (Bây giờ có 3 giá trị: Kênh, X, Y)
        c, x, y = image.shape
        
        # Nếu không đúng size 224x224 thì cắt gọt lại
        if x != self.output_size[0] or y != self.output_size[1]:
            # Chỉ zoom X và Y, kênh C tỷ lệ zoom là 1 (không zoom)
            image = zoom(image, (1, self.output_size[0] / x, self.output_size[1] / y), order=3)
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
            
        # Biến thành Tensor. 
        # (XÓA .unsqueeze(0) ở đây vì ảnh đã có 3 kênh rồi, không cần thêm kênh ảo nữa)
        image = torch.from_numpy(image.astype(np.float32))
        label = torch.from_numpy(label.astype(np.float32))
        
        sample = {'image': image, 'label': label.long()}
        return sample


class Synapse_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        self.data_dir = base_dir

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        # 1. Lấy tên file của lát cắt hiện tại (VD: "case0005_slice012")
        slice_name = self.sample_list[idx].strip('\n')
        data_path = os.path.join(self.data_dir, slice_name + '.npz')
        
        # 2. Load lát cắt hiện tại (Slice i)
        data = np.load(data_path)
        image_curr, label = data['image'], data['label']

        # ==== LOGIC 2.5D: TÌM LÁT CẮT TRƯỚC VÀ SAU ====
        image_prev = image_curr.copy() # Mặc định nếu không tìm thấy thì tự nhân đôi
        image_next = image_curr.copy()

        # Tách tên để lấy số thứ tự (Synapse thường có format: caseXXXX_sliceYYY)
        if "_slice" in slice_name:
            case_name, slice_idx_str = slice_name.split('_slice')
            slice_idx = int(slice_idx_str)
            
            # Tính toán tên file của lát trước (i-1) và lát sau (i+1)
            # (Định dạng %03d để nó ra 011, 012 thay vì 11, 12)
            prev_name = f"{case_name}_slice{slice_idx - 1:03d}"
            next_name = f"{case_name}_slice{slice_idx + 1:03d}"
            
            prev_path = os.path.join(self.data_dir, prev_name + '.npz')
            next_path = os.path.join(self.data_dir, next_name + '.npz')

            # Nếu file tồn tại (không bị kịch kim ở lát cắt đầu/cuối của bệnh nhân) thì load
            if os.path.exists(prev_path):
                image_prev = np.load(prev_path)['image']
            if os.path.exists(next_path):
                image_next = np.load(next_path)['image']
        # ===============================================

        # 3. Kẹp 3 lát cắt lại với nhau thành 1 khối (3, H, W)
        image = np.stack([image_prev, image_curr, image_next], axis=0)

        # Đóng gói dữ liệu
        sample = {'image': image, 'label': label}
        
        # 4. Ném vào Phân xưởng sơ chế (RandomGenerator) mà bạn vừa sửa ban nãy
        if self.transform:
            sample = self.transform(sample)
            
        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample