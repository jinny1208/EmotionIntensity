# import os
# import tarfile

# def process_root(root_dir, dry_run=True):
#     # Iterate through first-level subdirectories
#     for subfolder_name in os.listdir(root_dir):
#         subfolder_path = os.path.join(root_dir, subfolder_name)

#         if not os.path.isdir(subfolder_path):
#             continue

#         print(f"Processing: {subfolder_path}")

#         # 1️⃣ Delete existing .tar files
#         for item in os.listdir(subfolder_path):
#             item_path = os.path.join(subfolder_path, item)
#             if os.path.isfile(item_path) and item.endswith(".tar"):
#                 if dry_run:
#                     print(f"  [DRY RUN] Would delete: {item}")
#                 else:
#                     print(f"  Deleting: {item}")
#                     os.remove(item_path)

#         # 2️⃣ Tar remaining directories
#         for item in os.listdir(subfolder_path):
#             item_path = os.path.join(subfolder_path, item)

#             if os.path.isdir(item_path):
#                 tar_path = os.path.join(subfolder_path, f"{item}.tar")

#                 if dry_run:
#                     print(f"  [DRY RUN] Would create: {item}.tar")
#                 else:
#                     print(f"  Creating: {item}.tar")
#                     with tarfile.open(tar_path, "w") as tar:
#                         tar.add(item_path, arcname=item)


# if __name__ == "__main__":
#     root_directory = "/mnt/hdd/jeonyj0612/MEAD"

#     # 🔹 Set to True for dry run, False to actually execute
#     process_root(root_directory, dry_run=False)



################## STEP 2, after tar, delete unzipped folder:import os
import shutil
import os

def delete_untarred_folders(root_dir, dry_run=True):
    """
    Deletes only 'audio' and 'video' directories inside each first-level subfolder.
    Does NOT touch any .tar files.
    """

    for subfolder in os.listdir(root_dir):
        subfolder_path = os.path.join(root_dir, subfolder)

        if not os.path.isdir(subfolder_path):
            continue

        print(f"\nProcessing: {subfolder_path}")

        for target in ["audio", "video"]:
            target_path = os.path.join(subfolder_path, target)

            if os.path.isdir(target_path):
                if dry_run:
                    print(f"  [DRY RUN] Would delete folder: {target_path}")
                else:
                    print(f"  Deleting folder: {target_path}")
                    shutil.rmtree(target_path)
            else:
                print(f"  Skipping (not found): {target_path}")


if __name__ == "__main__":
    root_directory = "/mnt/hdd/jeonyj0612/MEAD"  # change if needed

    # 🔒 Dry run first
    delete_untarred_folders(root_directory, dry_run=False)
