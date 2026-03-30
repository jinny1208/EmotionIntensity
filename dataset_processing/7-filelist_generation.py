import os
import random
import glob
from collections import defaultdict

root = "/home/mila/j/jeony/scratch/MEAD_processed"

train_out = "/home/mila/j/jeony/EmotionIntensity/filelists/train_MEAD_filelist.txt"
val_out = "/home/mila/j/jeony/EmotionIntensity/filelists/val_MEAD_filelist.txt"
test_out = "/home/mila/j/jeony/EmotionIntensity/filelists/test_MEAD_filelist.txt"

random.seed(42)

# ---------------------------
# 1. collect wav files
# ---------------------------
groups = defaultdict(list)
all_files = []

for fname in os.listdir(root):

    if not fname.endswith(".wav"):
        continue

    wav_path = os.path.join(root, fname)
    all_files.append(wav_path)

    speaker, emotion, level, utt = fname.replace(".wav","").split("_")

    key = (speaker, emotion, utt)

    groups[key].append((level, wav_path))

# ---------------------------
# 2. classify groups
# ---------------------------
valid_groups = []
train_only = []

for key, items in groups.items():

    speaker, emotion, utt = key

    levels = {l for l,_ in items}
    wavs = [p for _,p in items]

    if emotion == "neutral":
        valid_groups.append(wavs)

    elif levels == {"level1","level2","level3"}:
        valid_groups.append(wavs)

    else:
        train_only.extend(wavs)

# ---------------------------
# 3. compute dataset targets
# ---------------------------
total_files = len(all_files)

val_target = int(total_files * 0.1)
test_target = int(total_files * 0.1)

# ---------------------------
# 4. split groups
# ---------------------------
random.shuffle(valid_groups)

train_files = train_only.copy()
val_files = []
test_files = []

for g in valid_groups:

    size = len(g)

    if len(val_files) + size <= val_target:
        val_files.extend(g)

    elif len(test_files) + size <= test_target:
        test_files.extend(g)

    else:
        train_files.extend(g)

# ---------------------------
# 5. build image index
# ---------------------------
image_index = defaultdict(list)

for fname in os.listdir(root):

    if not fname.endswith(".png"):
        continue

    speaker, emotion, level, *_ = fname.split("_")

    key = (speaker, emotion, level)

    image_index[key].append(os.path.join(root, fname))

# ---------------------------
# 6. helper functions
# ---------------------------

def find_transcript(wav_path):

    base = wav_path.replace(".wav","")

    corrected = base + "_CorrectedTranscript.txt"
    normal = base + ".txt"

    if os.path.exists(corrected):
        txt_path = corrected
    elif os.path.exists(normal):
        txt_path = normal
    else:
        return ""

    with open(txt_path,"r") as f:
        return f.read().strip()


def pick_image(wav_path):

    fname = os.path.basename(wav_path)

    speaker, emotion, level, utt = fname.replace(".wav","").split("_")

    key = (speaker, emotion, level)

    imgs = image_index.get(key, [])

    if not imgs:
        return ""

    return random.choice(imgs)

# ---------------------------
# 7. write filelist
# ---------------------------

def write_filelist(outfile, files):

    with open(outfile,"w") as f:

        for wav in files:

            img = pick_image(wav)
            txt = find_transcript(wav)

            line = f"{wav}|{img}|{txt}\n"

            f.write(line)

write_filelist(train_out, train_files)
write_filelist(val_out, val_files)
write_filelist(test_out, test_files)

# ---------------------------
# 8. report
# ---------------------------

print("Total:", total_files)
print("Train:", len(train_files))
print("Val:", len(val_files))
print("Test:", len(test_files))