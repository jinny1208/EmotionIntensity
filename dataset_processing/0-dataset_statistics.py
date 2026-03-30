# import os
# import re
# from collections import defaultdict

# DATA_DIR = "/home/mila/j/jeony/scratch/MEAD_processed"
# EXT = ".txt"

# pattern = re.compile(r"(?P<speaker>[MW]\d+)_(?P<emotion>\w+)_level(?P<intensity>\d+)")

# # ---------------------------------------------------
# # Scan dataset
# # ---------------------------------------------------
# speaker_groups = defaultdict(lambda: defaultdict(list))

# for fname in os.listdir(DATA_DIR):
#     if not fname.endswith(EXT):
#         continue

#     match = pattern.search(fname)
#     if not match:
#         continue

#     speaker = match.group("speaker")
#     emotion = match.group("emotion")
#     intensity = match.group("intensity")

#     key = (emotion, intensity)
#     speaker_groups[speaker][key].append(fname)

# # ---------------------------------------------------
# # Print counts per speaker
# # ---------------------------------------------------
# for speaker in sorted(speaker_groups.keys()):
#     print("\n=============================")
#     print(f"Speaker: {speaker}")
#     print("=============================")

#     groups = speaker_groups[speaker]

#     for (emotion, intensity), files in sorted(groups.items()):
#         print(f"{emotion} level{intensity}: {len(files)} files")


