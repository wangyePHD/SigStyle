import albumentations 
import os
import torch
import numpy as np
import pickle
import copy

from PIL import Image
from torch.utils.data import Dataset, DataLoader


def make_transforms(size, random_crop=False, ):
    rescaler = albumentations.SmallestMaxSize(max_size=size, interpolation=3)
    
    if not random_crop:
        cropper = albumentations.CenterCrop(height=size, width=size)
    else:
        cropper = albumentations.RandomCrop(height=size, width=size)
    flip = albumentations.HorizontalFlip(p=0.5)
    return albumentations.Compose([rescaler, cropper, flip])
    

def make_transforms_shape_inversion(size):
    resize = albumentations.Resize(size,size)

    return albumentations.Compose([resize])
    


class ShapeInvDataset(Dataset):
    def __init__(
            self,
            origin_img_path,
            resolution=512,
            is_need_aug=True,
    ):
        super().__init__()
        
        # load origin image
        if os.path.isfile(origin_img_path):
            with open(origin_img_path, 'r') as f:
                self.origin_img_list = [line.strip() for line in f.readlines()]
        
        if is_need_aug:
            self.processor = make_transforms(resolution, random_crop=True)
        else:
            self.processor = make_transforms_shape_inversion(resolution)
            

    def __len__(self):
        return len(self.origin_img_list)

    def __getitem__(self, idx):
        origin_img_path = self.origin_img_list[idx]
        # print(origin_img_path)
        image = Image.open(origin_img_path)
        image = np.array(image.convert("RGB"))
    
        transform = self.processor(image=image)
        image = transform["image"]
        
        image = (image / 127.5 - 1.0).astype(np.float32)
        image = torch.from_numpy(image).permute(2, 0, 1)
        
        
        return dict(
            pixel_values=image,
        )



class AppearanceInvDataset(Dataset):
    def __init__(
            self,
            origin_img_path,
            mask_img_path,
            appearance_img_path,
            resolution=512,
            is_need_aug=True,
    ):
        super().__init__()
        
        # load origin image
        if os.path.isfile(origin_img_path):
            with open(origin_img_path, 'r') as f:
                self.origin_img_list = [line.strip() for line in f.readlines()]
        # load mask image
        if os.path.isfile(mask_img_path):
            with open(mask_img_path, 'r') as f:
                self.mask_img_list = [line.strip() for line in f.readlines()]
        # load appearance image
        if os.path.isfile(appearance_img_path):
            with open(appearance_img_path, 'r') as f:
                self.app_img_list = [line.strip() for line in f.readlines()]
        
        if is_need_aug:
            self.processor = make_transforms(resolution, random_crop=True)
        else:
            self.processor = make_transforms_shape_inversion(resolution)
        

    def __len__(self):
        return len(self.origin_img_list)

    def __getitem__(self, idx):
        origin_img_path = self.origin_img_list[idx]
        mask_img_path = self.mask_img_list[idx]
        app_img_path = self.app_img_list[idx]
        
        image = Image.open(origin_img_path)
        image = np.array(image.convert("RGB"))
        
        app_img = Image.open(app_img_path)
        app_img = np.array(app_img.convert("RGB"))

        with open(mask_img_path, 'rb') as f:
            mask = pickle.load(f)
            mask[mask>0] = 1.0
            mask = np.expand_dims(np.array(mask),axis=-1)
            mask = np.concatenate([np.array(mask), np.array(mask), np.array(mask)], axis=2)
        
        image = image * mask  
        
        transform = self.processor(image=image, mask=mask)
        image = transform["image"]
        mask  = transform["mask"]
        
        transform_app = self.processor(image=app_img)
        app_img = transform_app["image"]

        mask = 255. * mask
        
        dino_image = image

        image = (image / 127.5 - 1.0).astype(np.float32)
        image = torch.from_numpy(image).permute(2, 0, 1)
        mask  = (mask / 127.5 - 1.0).astype(np.float32)
        mask  = torch.from_numpy(mask).permute(2, 0, 1)
        app_img = (app_img / 127.5 - 1.0).astype(np.float32)
        app_img = torch.from_numpy(app_img).permute(2, 0, 1)
        

        return dict(
            pixel_values=image,
            dino_image=dino_image,
            app_img=app_img,
            mask_values=mask,
        )

        
        
'''
if __name__ == '__main__':
    origin_img_path = "/home/wangye/Research2023/ShapeInversion/test_data/dogs/leopard.txt"
    mask_img_path = "/home/wangye/Research2023/ShapeInversion_Data/data/test_seg/leopard.txt"
    dataset = ShapeInvDataset(origin_img_path, mask_img_path, is_need_aug=False)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    for step, batch in enumerate(dataloader):
        print(batch['pixel_values'].shape)
        print(np.unique(batch['mask_values']))
        # # save batch['attn_mask'] as image using PIL
        # attn_mask = batch['attn_mask'][0].numpy()
        # print(attn_mask.shape)
        # attn_mask = Image.fromarray((attn_mask.astype(np.float32) * 255).astype(np.uint8))
        # attn_mask.save(f'./check_data/0/attn_mask_{step}.png')
        
        # attn_mask = batch['attn_mask1'][0].numpy()
        # print(attn_mask.shape)
        # attn_mask = Image.fromarray((attn_mask.astype(np.float32) * 255).astype(np.uint8))
        # attn_mask.save(f'./check_data/1/attn_mask_{step}.png')
        
        # save batch['pixel_values'] as image using PIL
        pixel_values = batch['pixel_values'][0].numpy()
        print(pixel_values.shape)
        pixel_values = Image.fromarray((pixel_values).astype(np.uint8))
        pixel_values.save(f'./test_data/leopard_no_bkg.png')
        
        # save batch["dino_image"] as image using PIL
        # dino_image = batch["dino_image"][0].numpy()
        # print(dino_image.shape)
        # dino_image = Image.fromarray((dino_image.astype(np.float32)).astype(np.uint8))
        # dino_image.save(f'./check_data/3/dino_image_{step}.png')
        break
'''