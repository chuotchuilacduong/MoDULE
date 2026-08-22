import os
import argparse

def download_office_home_hf(data_dir):
    """downloads and formats the office-home dataset using hugging face."""
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise RuntimeError(
            "this script requires the hugging face `datasets` package.\n"
            "install it with: pip install datasets pillow"
        ) from e

    print(f"Preparing to download Office-Home to: {data_dir}")
    
    # download from huggingface repo: flwrlabs/office-home
    print("Fetching dataset from Hugging Face Hub (flwrlabs/office-home)...")
    dataset = load_dataset("flwrlabs/office-home", split="train")
    
    # extract the actual string class names from the dataset features
    label_names = dataset.features["label"].names

    # set up the target directory
    os.makedirs(data_dir, exist_ok=True)

    print(f"Extracting and saving images to {data_dir}...")
    for i, sample in enumerate(dataset):
        domain = sample["domain"]
        class_name = label_names[sample["label"]]
        image = sample["image"]

        # folder structure: domain/class_name/image.jpg
        out_dir = os.path.join(data_dir, domain, class_name)
        os.makedirs(out_dir, exist_ok=True)
        
        out_file = os.path.join(out_dir, f"{i:06d}.jpg")
        
        # save the image locally 
        image.convert("RGB").save(out_file, format="JPEG", quality=100)

        # print progress every 2000 images
        if (i + 1) % 2000 == 0:
            print(f"Saved {i + 1} / {len(dataset)} images...")

    print(f"\nSuccess! Dataset is beautifully reconstructed at: {data_dir}")

if __name__ == "__main__":
    # target directory is fixed by default
    target_dir = "./dataset/data_folder/officehome"
    download_office_home_hf(target_dir)