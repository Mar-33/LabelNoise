from phenobench import PhenoBench
from pprint import pprint
from PIL import Image
import ipdb
import torch
import numpy as np

def leaf_noise(masks, leafs, new_class, leaf_noise_factor, device):
  if leaf_noise_factor == 0:
    return masks
  else:
    # leaf = torch.tensor(leafs).long().to(device)
    for i, mask in enumerate(masks):
      leaf_ids = torch.unique(leafs[i]).cpu().detach().numpy()
      if leaf_ids.size > 1:
        random_instance = np.random.choice(leaf_ids[1:],int(np.ceil(len(leaf_ids)*leaf_noise_factor)))
        masks[i][ torch.isin(leafs[i],torch.tensor(random_instance))] = new_class
    return masks

class Dataset(object):
    def __init__(self, root, transform=None, split=None, data=None, leaf_instances = False, leaf_class = None, leaf_noise_factor = None, device = None):
        self.transform = transform
        self.root = root
        self.split = split
        # leaf stuff:
        self.leaf_instances = leaf_instances
        self.leaf_class = leaf_class
        self.leaf_noise_factor = leaf_noise_factor
        self.device = device
        if self.leaf_instances:
           self.data = PhenoBench(self.root, target_types=["semantics", 'leaf_instances'], split = self.split)
        else: self.data = PhenoBench(self.root, target_types=["semantics"], split = self.split)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # self.data = PhenoBench(self.root, target_types=["semantics"], split = self.split) #target_types=["semantics", "plant_instances", "leaf_instances"]

        image = self.data[idx]['image']
        mask = self.data[idx]['semantics']
        mask[mask > 2] -= 2
        mask = Image.fromarray(mask)
        if self.leaf_instances:
          leaf = Image.fromarray(self.data[idx]['leaf_instances'])

        if self.transform:
            image = self.transform['image'](image)
            mask = self.transform['mask'](mask)
            if self.leaf_instances:
              leaf = self.transform['mask'](leaf)
              mask = leaf_noise(mask, leaf, new_class = self.leaf_class , leaf_noise_factor = self.leaf_noise_factor, device = self.device)

        if self.leaf_instances:
          return image, mask, leaf
        else: return image, mask, torch.tensor([])
