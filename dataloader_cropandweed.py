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

    
# Classes:  
# 0: 'Maize'
# 1: 'Sugar beet'
# 2: 'Soy'
# 3: 'Sunflower'
# 4: 'Potato'
# 5: 'Pea'
# 6: 'Bean'
# 7: 'Pumpkin'
# 8: 'Weed'
# 9: 'Soil'

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # Load Image
        image = Image.open(self.img_paths[idx])
        # Load Mask and correct classes
        mask = imageio.imread(self.mask_paths[idx])
        ipdb.set_trace()
        mask += 1
        mask[mask == 10] = 0
        # change class values only to classes soil [0], plant [1], weed [2]
        mask[(mask > 0) & (mask < 9)] = 1
        mask[mask == 9] = 2
        ipdb.set_trace()

# Classes:  0: Soil   1: Maize   2: Sugar Beet   3: Soy       4: Sunflower   
#           5: Potato 6: Pea     7: Bean         8: Pumpkin   9: Weed

        if self.transform:
            image = self.transform['image'](image)
            mask = Image.fromarray(mask)
            mask = self.transform['mask'](mask)
        return image, mask
