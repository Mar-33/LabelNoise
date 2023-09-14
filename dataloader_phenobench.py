from phenobench import PhenoBench
from pprint import pprint
from PIL import Image

class PhenoBenchDataset(object):
    def __init__(self, root, transform=None, split=None, data=None):
        self.transform = transform
        self.root = root
        self.split = split
        self.data = PhenoBench(self.root, target_types=["semantics"], split = self.split)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # self.data = PhenoBench(self.root, target_types=["semantics"], split = self.split) #target_types=["semantics", "plant_instances", "leaf_instances"]

        image = self.data[idx]['image']
        mask = Image.fromarray(self.data[idx]['semantics'])

        if self.transform:
            image = self.transform['image'](image)
            mask = self.transform['mask'](mask)

        return image, mask
