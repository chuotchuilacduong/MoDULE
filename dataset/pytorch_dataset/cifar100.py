import os
import torch
from PIL import Image
from torch.utils.data import Dataset

class CIFAR100Dataset(Dataset):
    """custom pytorch dataset to parse the split/class/image structure of cifar-100."""
    
    def __init__(self, root_dir, split="train", transform=None):
        """
        args:
            root_dir (str): the base directory where cifar100 was downloaded (e.g., './dataset/data_folder/cifar100').
            split (str): which split to load ('train' or 'test').
            transform (callable, optional): optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        
        self.image_paths = []
        self.labels = []
        
        # target the specific split directory
        self.split_dir = os.path.join(root_dir, split)
        if not os.path.exists(self.split_dir):
            raise RuntimeError(f"split directory not found: {self.split_dir}")

        # extract class names (the 100 fine classes)
        self.class_names = sorted([c for c in os.listdir(self.split_dir) if os.path.isdir(os.path.join(self.split_dir, c))])
        self.class_to_idx = {c: i for i, c in enumerate(self.class_names)}

        # traverse the folder structure and catalog every image
        for class_name in self.class_names:
            class_idx = self.class_to_idx[class_name]
            class_dir = os.path.join(self.split_dir, class_name)
            
            for img_name in sorted(os.listdir(class_dir)):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(class_dir, img_name))
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
        return image, label

# # --- Example Usage ---
# if __name__ == "__main__":
#     import torchvision.transforms as T
#     from torch.utils.data import DataLoader

#     # standard cifar100 transforms (keeping images at 32x32)
#     transform = T.Compose([
#         T.ToTensor(),
#         T.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
#     ])

#     data_dir = "./dataset/data_folder/cifar100"

#     # load the datasets
#     train_dataset = CIFAR100Dataset(root_dir=data_dir, split="train", transform=transform)
#     test_dataset = CIFAR100Dataset(root_dir=data_dir, split="test", transform=transform)

#     # create dataloaders
#     train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
#     test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

#     print(f"Total training images: {len(train_dataset)}")
#     print(f"Total testing images: {len(test_dataset)}")
    
#     # check shape of first batch
#     for images, labels in train_loader:
#         print(f"Image batch shape: {images.shape}")
#         print(f"Labels batch shape: {labels.shape}")
#         break