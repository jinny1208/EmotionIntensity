import os

input_file = "/home/mila/j/jeony/EmotionIntensity/filelists/val_MEAD_filelist_espeak.txt"
output_file = "/home/mila/j/jeony/EmotionIntensity/filelists/val_MEAD_filelist_espeak_with_attrs.txt"

with open(input_file, "r") as fin, open(output_file, "w") as fout:
    for line in fin:
        line = line.strip()
        if not line:
            continue

        parts = line.split("|")
        wav_path = parts[0]

        # Extract filename
        fname = os.path.basename(wav_path)  # M019_angry_level2_027.wav
        tokens = fname.replace(".wav", "").split("_")

        # Expected format: ID_emotion_levelX_###
        emotion = tokens[1]
        level = tokens[2]

        # Append attributes
        new_line = line + f"|{emotion}|{level}"
        fout.write(new_line + "\n")