import os
import argparse

def download_pacs_hf(full_path):
    """downloads pacs directly from hugging face and organizes by domain and class."""
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise RuntimeError(
            "hugging face download requires the `datasets` and `pillow` packages.\n"
            "install them with: pip install datasets pillow"
        ) from e

    if os.path.isdir(full_path) and any(os.scandir(full_path)):
        print(f"PACS already exists at {full_path}, skipping download.")
        return

    print("Fetching dataset from hugging face hub (flwrlabs/pacs)...")
    # download from flwrlabs/pacs repo
    dataset = load_dataset("flwrlabs/pacs", split="train")
    label_names = dataset.features["label"].names

    os.makedirs(full_path, exist_ok=True)
    print(f"Extracting and saving images to {full_path}...")
    
    for i, sample in enumerate(dataset):
        domain = sample["domain"]
        class_name = label_names[sample["label"]]
        image = sample["image"]

        # build standard folder structure: domain/class_name/image.jpg
        out_dir = os.path.join(full_path, domain, class_name)
        os.makedirs(out_dir, exist_ok=True)
        
        out_file = os.path.join(out_dir, f"{i:06d}.jpg")
        
        # save image locally as jpeg
        image.convert("RGB").save(out_file, format="JPEG", quality=95)
        
        if (i + 1) % 1000 == 0:
            print(f"Saved {i + 1} / {len(dataset)} images...")

    print(f"Success! dataset is ready at: {full_path}")

if __name__ == "__main__":
    # target directory is fixed by default
    target_dir = "./dataset/data_folder/pacs"
    
    # ensure the parent directory exists
    os.makedirs(os.path.dirname(target_dir), exist_ok=True)
    
    download_pacs_hf(target_dir)