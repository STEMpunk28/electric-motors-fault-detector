import os
import shutil
import random

def split_dataset(input_dir, output_dir, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=42):
    random.seed(seed)

    # Get all files
    files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
    random.shuffle(files)

    # Compute split indices
    n_total = len(files)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_files = files[:n_train]
    val_files = files[n_train:n_train+n_val]
    test_files = files[n_train+n_val:]

    # Create output dirs
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    test_dir = os.path.join(output_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Move/copy files
    def copy_files(file_list, dest_dir):
        for f in file_list:
            shutil.copy2(os.path.join(input_dir, f), os.path.join(dest_dir, f))

    copy_files(train_files, train_dir)
    copy_files(val_files, val_dir)
    copy_files(test_files, test_dir)

    print(f"Total files: {n_total}")
    print(f"Train: {len(train_files)} → {train_dir}")
    print(f"Val:   {len(val_files)} → {val_dir}")
    print(f"Test:  {len(test_files)} → {test_dir}")

# Example usage
if __name__ == "__main__":
    input_folder = "D:/TrainingSet/Abnormal"
    output_folder = "D:/TrainingSet/Abnormal"
    split_dataset(input_folder, output_folder)
