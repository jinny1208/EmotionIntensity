import re
import os

def get_high_wer_indices(file_a, threshold=0.4):
    high_wer_indices = set()

    with open(file_a, "r", encoding="utf-8") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if line.endswith(".wav"):
            wav_id = line.replace(".wav", "")

            if i + 3 < len(lines):
                wer_line = lines[i + 3].strip()
                match = re.search(r"WER:\s*([0-9.]+)", wer_line)

                if match:
                    wer = float(match.group(1))
                    if wer > threshold:
                        high_wer_indices.add(wav_id)

            i += 1
        else:
            i += 1

    return high_wer_indices


def extract_index_from_path(path):
    """
    Extract index like 00001 from filenames if present.
    Example:
        W024_happy_level1_001.wav -> 001 (or adapt if needed)
    """
    filename = os.path.basename(path)
    
    # Try to match trailing numbers before .wav
    match = re.search(r"(\d+)\.wav$", filename)
    if match:
        return match.group(1).zfill(5)  # normalize to 5 digits

    return None


def filter_file_b(file_b, output_b, remove_indices):
    with open(file_b, "r", encoding="utf-8") as f:
        lines = f.readlines()

    filtered_lines = []

    for line in lines:
        line = line.rstrip("\n")

        if not line.strip():
            continue

        parts = line.split("|")
        if len(parts) < 1:
            continue

        wav_path = parts[0]
        file_index = extract_index_from_path(wav_path)

        if file_index is None:
            # keep if we cannot parse
            filtered_lines.append(line + "\n")
            continue

        if file_index in remove_indices:
            # skip this line
            continue

        filtered_lines.append(line + "\n")

    with open(output_b, "w", encoding="utf-8") as f:
        f.writelines(filtered_lines)


if __name__ == "__main__":
    file_a = "/home/mila/j/jeony/scratch/EmotionIntensityClassifier/logs/0-vctk_base_wer_results_onVCTKtextandspk.txt"
    file_b = "/home/mila/j/jeony/EmotionIntensity/filelists/final_output_test_VCTK_onVCTKtextandspk.txt"
    output_b = "/home/mila/j/jeony/EmotionIntensity/filelists/final_output_test_VCTK_onVCTKtextandspk_linesRemoved.txt"

    remove_indices = get_high_wer_indices(file_a, threshold=0.4)

    print(f"Removing indices: {remove_indices}")

    filter_file_b(file_b, output_b, remove_indices)

    print(f"Saved filtered file to: {output_b}")