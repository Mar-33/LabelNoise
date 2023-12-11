from phenobench import PhenoBench
from pprint import pprint
from PIL import Image
import imageio
import torch
import glob
import os
import ipdb

class Dataset(object):
    def __init__(self, root, transform=None, split=None, data=None, leaf_instances = False, leaf_class = None, leaf_noise_factor = None, device = None):
        self.transform = transform
        self.root = root
        self.split = split
        self.leaf_instances = leaf_instances
        self.img_paths = glob.glob(os.path.join(self.root, self.split, 'images', '*.jpg'))
        self.img_paths.sort()
        self.mask_paths = glob.glob(os.path.join(self.root, self.split, 'CropsOrWeed9', '*.png'))
        self.mask_paths.sort()
        self.leaf_instances = False

    
# Classes:  0: Soil   1: Maize   2: Sugar Beet   3: Soy       4: Sunflower   
#           5: Potato 6: Pea     7: Bean         8: Pumpkin   9: Weed

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # Load Image
        image = Image.open(self.img_paths[idx])
        # Load Mask and correct classes
        mask = imageio.imread(self.mask_paths[idx])
        mask += 1
        mask[mask == 10] = 0

        if self.transform:
            image = self.transform['image'](image)
            mask = Image.fromarray(mask)
            mask = self.transform['mask'](mask)
        return image, mask