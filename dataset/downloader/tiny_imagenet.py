import os
import urllib.request
import argparse
from zipfile import ZipFile

def download_and_extract(url, dst, remove=True):
    """downloads a file from a direct url and extracts it if it is a zip archive."""
    print(f"Downloading from {url} to {dst}...")
    
    # using standard urllib since this is a direct http link, no gdown required
    urllib.request.urlretrieve(url, dst)

    if dst.endswith(".zip"):
        print("Extracting zip archive...")
        zf = ZipFile(dst, "r")
        zf.extractall(os.path.dirname(dst))
        zf.close()

    if remove:
        print("Cleaning up zip file...")
        os.remove(dst)

def download_tiny_imagenet(data_dir):
    """downloads, extracts, and organizes the tiny imagenet dataset."""
    os.makedirs(data_dir, exist_ok=True)
    full_path = os.path.join(data_dir, "tiny-imagenet-200")

    # check if it already exists
    if os.path.isdir(full_path) and any(os.scandir(full_path)):
        print(f"Tiny ImageNet already exists at {full_path}, skipping download.")
        return

    # stanford's official download link
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    zip_dst = os.path.join(data_dir, "tiny-imagenet-200.zip")

    try:
        download_and_extract(url, zip_dst)
    except Exception as e:
        print(f"download failed: {e}")
        return

    # reorganize validation data into class folders 
    val_dir = os.path.join(full_path, 'val')
    val_images_dir = os.path.join(val_dir, 'images')
    val_annotations_file = os.path.join(val_dir, 'val_annotations.txt')
    
    if os.path.exists(val_annotations_file):
        print("Reorganizing validation set into class subdirectories...")
        with open(val_annotations_file, 'r') as f:
            val_data = f.readlines()
            
        for line in val_data:
            parts = line.strip().split('\t')
            img_name, class_id = parts[0], parts[1]
            
            # create class folder inside val directory
            class_dir = os.path.join(val_dir, class_id)
            os.makedirs(class_dir, exist_ok=True)
            
            # move image to its respective class folder
            src = os.path.join(val_images_dir, img_name)
            dst = os.path.join(class_dir, img_name)
            if os.path.exists(src):
                os.rename(src, dst)
        
        # clean up the now-empty images directory and the annotations file
        if os.path.exists(val_images_dir) and not os.listdir(val_images_dir):
            os.rmdir(val_images_dir)
        os.remove(val_annotations_file)

    print(f"Success! dataset is reconstructed at: {full_path}")

if __name__ == "__main__":
    # target directory is fixed by default
    target_dir = "./dataset/data_folder/tiny_imagenet"
    download_tiny_imagenet(target_dir)