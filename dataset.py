from library import *
# from image_transform import *


# MAKE DATASET
class MyDataset(data.Dataset):
    def __init__(self, file_list, transform=None, phase="train"):
        self.file_list = file_list
        self.transform = transform
        self.phase = phase
        super().__init__()

    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path)

        img_transformed = self.transform(img, self.phase)

        if self.phase == "train":
            label = img_path[43:49]
            if label == "female":
                label = 0
            else:
                label  = img_path[43:47]
                label = 1

        elif self.phase == "val":
            label = img_path[41:47]
            if label == "female":
                label = 0
            else:
                label  = img_path[41:45]
                label = 1

        return img_transformed, label
