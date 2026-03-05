### STEP 1: Find the problematic groups (there should be a total of 8 files for each utterance ID)

# import os
# from collections import defaultdict

# dataset_dir = "/home/mila/j/jeony/scratch/MEAD_processed"  # change this
# expected_count = 8

# groups = defaultdict(list)

# # scan files
# for fname in os.listdir(dataset_dir):
#     if fname.endswith((".txt", ".wav", ".png")):
#         # remove extension
#         base = os.path.splitext(fname)[0]

#         # remove _fgX if present
#         if "_fg" in base:
#             base = base.split("_fg")[0]

#         groups[base].append(fname)

# errors = []

# # check counts
# for base, files in groups.items():
#     if len(files) != expected_count:
#         errors.append(base)

# # write errors
# with open("file_grouping_errors.txt", "w") as f:
#     for e in errors:
#         f.write(e + "\n")

# print(f"Checked {len(groups)} groups.")
# print(f"Found {len(errors)} grouping errors.")




#### STEP 2: delete the problematic groups as well as their other emotion intensities level
import os
import pdb

dataset_root = "/home/mila/j/jeony/scratch/MEAD_processed"
filelist = "/home/mila/j/jeony/scratch/MEAD_processed/file_grouping_errors.txt"
dry_run_output = "/home/mila/j/jeony/scratch/MEAD_processed/files_dry_run2.txt"

dry_run = False

# read target groups
targets = set()
with open(filelist) as f:
    for line in f:
        name = line.strip()
        speaker, emotion, level, utt = name.split("_")
        targets.add((speaker, emotion, utt))

files_to_delete = []

# single directory scan
for fname in os.listdir(dataset_root):

    base = os.path.splitext(fname)[0]

    if "_fg" in base:
        base = base.split("_fg")[0]
        # pdb.set_trace()

    parts = base.split("_")
    if len(parts) != 4:
        continue

    speaker, emotion, level, utt = parts

    if (speaker, emotion, utt) in targets:
        files_to_delete.append(os.path.join(dataset_root, fname))

print("Matched files:", len(files_to_delete))

if dry_run:
    with open(dry_run_output, "w") as out:
        for f in files_to_delete:
            out.write(f + "\n")
    print(f"Dry run results saved to: {dry_run_output}")

else:
    for f in files_to_delete:
        os.remove(f)
    print("Files deleted.")