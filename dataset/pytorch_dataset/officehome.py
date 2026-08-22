import os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as T
import random

class OfficeHomeDataset(Dataset):
    """custom pytorch dataset to parse the domain/class/image structure of office-home."""
    
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        self.image_paths = []
        self.labels = []
        self.domains = []

        # extract domain names (art, clipart, product, real_world)
        self.domain_names = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.domain_to_idx = {d: i for i, d in enumerate(self.domain_names)}

        # extract class names (assuming all domains share the 65 classes)
        first_domain = os.path.join(root_dir, self.domain_names[0])
        self.class_names = sorted([c for c in os.listdir(first_domain) if os.path.isdir(os.path.join(first_domain, c))])
        self.class_to_idx = {c: i for i, c in enumerate(self.class_names)}

        # traverse the folder structure and catalog every image
        for domain in self.domain_names:
            domain_idx = self.domain_to_idx[domain]
            for class_name in self.class_names:
                class_idx = self.class_to_idx[class_name]
                class_dir = os.path.join(root_dir, domain, class_name)
                
                if not os.path.exists(class_dir):
                    continue
                    
                for img_name in sorted(os.listdir(class_dir)):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.image_paths.append(os.path.join(class_dir, img_name))
                        self.labels.append(class_idx)
                        self.domains.append(domain_idx)

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
        domain = self.domains[idx]

        return image, label, domain

# def get_officehome_unlearning_dataloaders(data_dir, batch_size=32, num_workers=4, shuffle_before_split=True):
#     """splits the office-home dataset explicitly using subsets to guarantee no overlap."""
    
#     # standard resnet/imagenet transforms
#     transform = T.Compose([
#         T.Resize((224, 224)),
#         T.ToTensor(),
#         T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])

#     full_dataset = OfficeHomeDataset(root_dir=data_dir, transform=transform)
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
#     # path to where the office-home script downloaded the images
#     officehome_dir = "./dataset/data_folder/officehome"
    
#     # generate the loaders
#     train_loader, test_loader, unseen_loader = get_officehome_unlearning_dataloaders(officehome_dir, batch_size=32)
    
#     # verify the logic
#     print(f"Total batches in Train Loader: {len(train_loader)}")
#     print(f"Total batches in Test Loader: {len(test_loader)}")
#     print(f"Total batches in Unseen Loader: {len(unseen_loader)}")
    
#     # check shape of first batch
#     for images, labels, domains in train_loader:
#         print(f"Image batch shape: {images.shape}")
#         print(f"Labels batch shape: {labels.shape}")
#         print(f"Domains batch shape: {domains.shape}")
#         break