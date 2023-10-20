from phenobench import PhenoBench
from pprint import pprint
from PIL import Image

class PhenoBenchDataset(object):
    def __init__(self, root, transform=None, split=None, data=None, leaf_instances = False):
        self.transform = transform
        self.root = root
        self.split = split
        self.leaf_instances = leaf_instances
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

        if self.leaf_instances:
          return image, mask, leaf
        else: return image, mask
