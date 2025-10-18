import os
import random
import shutil



def copy_random_folders(source_path, destination_path, num_folders=1000):
    all_dirs = []
    for d in os.listdir(source_path):
        folder_path = os.path.join(source_path, d)
        if os.path.isdir(folder_path):
            items = os.listdir(folder_path)
            if len(items) >= 100:
                all_dirs.append(d)

    if len(all_dirs) < num_folders:
        raise ValueError(
            f"Folder with at least 100 files has {len(all_dirs)} folders, so it is not possible to copy {num_folders} folders."
        )

    selected_dirs = random.sample(all_dirs, num_folders)

    os.makedirs(destination_path, exist_ok=True)
    for folder_name in selected_dirs:
        src = os.path.join(source_path, folder_name)
        dst = os.path.join(destination_path, folder_name)

        if os.path.exists(dst):
            print(f"Destination folder already exists: {dst}")
            continue

        print(f"Copying: {src} -> {dst}")
        shutil.copytree(src, dst)

def split_each_class_by_ratio(source_path, destination_path, test_ratio=0.1):
    class_folders = [
        d for d in os.listdir(source_path)
        if os.path.isdir(os.path.join(source_path, d))
    ]

    train_root = os.path.join(destination_path, "train")
    test_root = os.path.join(destination_path, "test")
    os.makedirs(train_root, exist_ok=True)
    os.makedirs(test_root, exist_ok=True)

    for class_folder in class_folders:
        class_path = os.path.join(source_path, class_folder)

        all_files = [
            f for f in os.listdir(class_path)
            if os.path.isfile(os.path.join(class_path, f))
        ]
        if not all_files:
            print(f"Warning: {class_folder} folder has no files.")
            continue

        test_count = int(len(all_files) * test_ratio)
        test_files = random.sample(all_files, test_count)
        train_files = set(all_files) - set(test_files)

        train_class_path = os.path.join(train_root, class_folder)
        test_class_path = os.path.join(test_root, class_folder)
        os.makedirs(train_class_path, exist_ok=True)
        os.makedirs(test_class_path, exist_ok=True)

        for file_name in test_files:
            src = os.path.join(class_path, file_name)
            dst = os.path.join(test_class_path, file_name)
            shutil.copy2(src, dst)

        for file_name in train_files:
            src = os.path.join(class_path, file_name)
            dst = os.path.join(train_class_path, file_name)
            shutil.copy2(src, dst)

        print(f"[{class_folder}] -> "
              f"train: {len(train_files)} files, test: {len(test_files)} files copied.")


if __name__ == "__main__":
    random.seed(42)

    source_path = "/home/compu/Datasets/youtube_faces/images"
    destination_path = "/home/compu/Datasets/youtube_faces_2000/images/all"
    copy_random_folders(source_path, destination_path, num_folders=2000)

    source_path = "/home/compu/Datasets/youtube_faces_2000/images/all"
    destination_path = "/home/compu/Datasets/youtube_faces_2000/images"

    split_each_class_by_ratio(
        source_path=source_path,
        destination_path=destination_path,
        test_ratio=0.2
    )