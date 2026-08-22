import os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T

class PACSDataset(Dataset):
    """custom pytorch dataset to parse the domain/class/image structure of pacs."""
    
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        self.image_paths = []
        self.labels = []
        self.domains = []

        # extract domain names (art_painting, cartoon, photo, sketch)
        self.domain_names = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.domain_to_idx = {d: i for i, d in enumerate(self.domain_names)}

        # extract class names (assuming all domains share the same classes)
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
                    
                for img_name in os.listdir(class_dir):
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

# def get_unlearning_dataloaders(data_dir, batch_size=32, num_workers=4):
#     """splits the pacs dataset into 60% train, 20% test, 20% unseen."""
    
#     # standard resnet/imagenet transforms
#     transform = T.Compose([
#         T.Resize((224, 224)),
#         T.ToTensor(),
#         T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])

#     # instantiate full dataset
#     full_dataset = PACSDataset(root_dir=data_dir, transform=transform)
#     total_size = len(full_dataset)

#     # calculate exact sizes for 60/20/20 split
#     train_size = int(0.6 * total_size)
#     test_size = int(0.2 * total_size)
#     unseen_size = total_size - train_size - test_size

#     # use a fixed manual seed so your splits remain identical across runs
#     generator = torch.Generator().manual_seed(42)
    
#     train_dataset, test_dataset, unseen_dataset = random_split(
#         full_dataset, 
#         [train_size, test_size, unseen_size], 
#         generator=generator
#     )

#     # create standard dataloaders
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
#     # testing/unseen loaders do not need to be shuffled
#     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
#     unseen_loader = DataLoader(unseen_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

#     return train_loader, test_loader, unseen_loader

# if __name__ == "__main__":
#     # path to where the previous script downloaded the images
#     pacs_dir = "./dataset/data_folder/pacs"
    
#     # generate the loaders
#     train_loader, test_loader, unseen_loader = get_unlearning_dataloaders(pacs_dir, batch_size=32)
    
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