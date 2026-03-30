import os
import re
import random
from collections import defaultdict

DATA_DIR = "/home/mila/j/jeony/scratch/MEAD_processed"
EXT = ".png"

dry_run = False   # True = preview only, False = actually delete
KEEP_PER_GROUP = 10

pattern = re.compile(r"(?P<speaker>[MW]\d+)_(?P<emotion>\w+)_level(?P<intensity>\d+)")

# ---------------------------------------------------
# Scan dataset
# ---------------------------------------------------

speaker_groups = defaultdict(lambda: defaultdict(list))

for fname in os.listdir(DATA_DIR):
    if not fname.endswith(EXT):
        continue

    match = pattern.search(fname)
    if not match:
        continue

    speaker = match.group("speaker")
    emotion = match.group("emotion")
    intensity = match.group("intensity")

    key = (emotion, intensity)

    speaker_groups[speaker][key].append(fname)

# ---------------------------------------------------
# Process speaker by speaker
# ---------------------------------------------------

all_keep = []
all_remove = []

for speaker in sorted(speaker_groups.keys()):

    print("\n=============================")
    print(f"Speaker: {speaker}")
    print("=============================")

    groups = speaker_groups[speaker]

    for (emotion, intensity), files in sorted(groups.items()):

        n_files = len(files)
        n_keep = min(KEEP_PER_GROUP, n_files)

        print(f"{emotion} level{intensity}: {n_files} files → keeping {n_keep}")

        selected = random.sample(files, n_keep)

        for f in files:

            fullpath = os.path.join(DATA_DIR, f)

            if f in selected:
                all_keep.append(fullpath)
            else:
                all_remove.append(fullpath)

# ---------------------------------------------------
# Summary
# ---------------------------------------------------

print("\n=============================")
print("FINAL SUMMARY")
print("=============================")

print(f"Total files: {len(all_keep) + len(all_remove)}")
print(f"Keep: {len(all_keep)}")
print(f"Remove: {len(all_remove)}")

print("\nExample KEEP:")
for f in all_keep[:10]:
    print("KEEP ", f)

print("\nExample REMOVE:")
for f in all_remove[:10]:
    print("DROP ", f)

# save lists
with open("/home/mila/j/jeony/EmotionIntensity/dataset_processing/keep_list.txt", "w") as f:
    for k in all_keep:
        f.write(k + "\n")

with open("/home/mila/j/jeony/EmotionIntensity/dataset_processing/remove_list.txt", "w") as f:
    for r in all_remove:
        f.write(r + "\n")

print("\nLists saved: keep_list.txt / remove_list.txt")

# ---------------------------------------------------
# Apply deletion
# ---------------------------------------------------

if not dry_run:

    print("\nDeleting files...")

    for f in all_remove:
        os.remove(f)

    print(f"Deleted {len(all_remove)} files.")

else:

    print("\nDRY RUN ENABLED — no files deleted.")