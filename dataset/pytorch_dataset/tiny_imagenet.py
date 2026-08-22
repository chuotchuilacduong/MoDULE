import os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as T
import random

class TinyImageNetDataset(Dataset):
    """custom pytorch dataset to parse the unique structure of tiny imagenet."""
    
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        self.image_paths = []
        self.labels = []

        train_dir = os.path.join(root_dir, "train")
        val_dir = os.path.join(root_dir, "val")

        # extract the 200 class names from the train directory
        self.class_names = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
        self.class_to_idx = {c: i for i, c in enumerate(self.class_names)}

        # 1. traverse the train folder structure (train/class_id/images/img.JPEG)
        for class_name in self.class_names:
            class_idx = self.class_to_idx[class_name]
            # notice the extra 'images' subfolder for the training set
            class_img_dir = os.path.join(train_dir, class_name, "images")
            
            if os.path.exists(class_img_dir):
                for img_name in sorted(os.listdir(class_img_dir)):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.image_paths.append(os.path.join(class_img_dir, img_name))
                        self.labels.append(class_idx)

        # 2. traverse the val folder structure (val/class_id/img.JPEG)
        # (this relies on the reorganization done by the download script)
        for class_name in self.class_names:
            class_idx = self.class_to_idx[class_name]
            class_img_dir = os.path.join(val_dir, class_name)
            
            if os.path.exists(class_img_dir):
                for img_name in sorted(os.listdir(class_img_dir)):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.image_paths.append(os.path.join(class_img_dir, img_name))
                        self.labels.append(class_idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # load image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        
        # apply transforms if any
        if self.transform:
            image = self.transform(image)
            
        label = self.labels[idx]

        # tiny imagenet only requires image and label (no domain)
        return image, label

# def get_tinyimagenet_unlearning_dataloaders(data_dir, batch_size=32, num_workers=4, shuffle_before_split=True):
#     """splits the tiny imagenet dataset explicitly using subsets to guarantee no overlap."""
    
#     # standard resnet transforms 
#     # (tiny imagenet is natively 64x64, but we resize to 224x224 to match standard models)
#     transform = T.Compose([
#         T.Resize((224, 224)),
#         T.ToTensor(),
#         T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])

#     full_dataset = TinyImageNetDataset(root_dir=data_dir, transform=transform)
#     total_size = len(full_dataset)

#     # create a list of all absolute indices
#     indices = list(range(total_size))

#     # optionally shuffle the indices globally before slicing 
#     if shuffle_before_split:
#         # fixed seed ensures identical splits every time the script runs
#         random.seed(42)
#         random.shuffle(indices)

#     # calculate exact index boundaries for a 60 / 20 / 20 split
#     train_bound = int(0.6 * total_size)
#     test_bound = int(0.8 * total_size)

#     # slice the indices explicitly
#     train_indices = indices[:train_bound]
#     test_indices = indices[train_bound:test_bound]
#     unseen_indices = indices[test_bound:]

#     # create mutually exclusive subsets
#     train_dataset = Subset(full_dataset, train_indices)
#     test_dataset = Subset(full_dataset, test_indices)
#     unseen_dataset = Subset(full_dataset, unseen_indices)

#     # create dataloaders
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
#     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
#     unseen_loader = DataLoader(unseen_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

#     # safety check: mathematically prove there is zero intersection
#     train_set = set(train_indices)
#     test_set = set(test_indices)
#     unseen_set = set(unseen_indices)
    
#     assert train_set.isdisjoint(test_set), "train and test sets intersect!"
#     assert train_set.isdisjoint(unseen_set), "train and unseen sets intersect!"
#     assert test_set.isdisjoint(unseen_set), "test and unseen sets intersect!"

#     return train_loader, test_loader, unseen_loader

# if __name__ == "__main__":
#     # path to where the tiny imagenet script downloaded the images
#     # it targets the root of the extracted folder containing 'train' and 'val'
#     tinyimagenet_dir = "./dataset/data_folder/tiny_imagenet/tiny-imagenet-200"
    
#     # generate the loaders
#     train_loader, test_loader, unseen_loader = get_tinyimagenet_unlearning_dataloaders(tinyimagenet_dir, batch_size=32)
    
#     # verify the logic
#     print(f"Total batches in Train Loader: {len(train_loader)}")
#     print(f"Total batches in Test Loader: {len(test_loader)}")
#     print(f"Total batches in Unseen Loader: {len(unseen_loader)}")
    
#     # check shape of first batch
#     for images, labels in train_loader:
#         print(f"Image batch shape: {images.shape}")
#         print(f"Labels batch shape: {labels.shape}")
#         break