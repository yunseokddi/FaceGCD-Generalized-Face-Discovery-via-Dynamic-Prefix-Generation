import os

dir_path = 'gcd_youtubefaces_1000_part_fvit_pretrain_prefix10/train'
folders_to_rename = []

for folder_name in os.listdir(dir_path):
    folder_path = os.path.join(dir_path, folder_name)

    if folder_name.isdigit() and os.path.isdir(folder_path):
        new_name = str(int(folder_name) - 1)

        old_folder = os.path.join(dir_path, folder_name)
        new_folder = os.path.join(dir_path, new_name)

        folders_to_rename.append((old_folder, new_folder))

for old_folder, new_folder in folders_to_rename:
    os.rename(old_folder, new_folder)

print("Folder name change completed.")