file_a = "/home/mila/j/jeony/EmotionIntensity/filelists/all_vctk_text_org.txt"
file_b = "/home/mila/j/jeony/EmotionIntensity/filelists/all_vctk_text_org.txt.cleaned"
file_c = "/home/mila/j/jeony/EmotionIntensity/filelists/test_VCTK_onVCTKtextandspk.txt"

output_file = "/home/mila/j/jeony/EmotionIntensity/filelists/final_output_test_VCTK_onVCTKtextandspk.txt"


# ===== LOAD FILE A (2nd column) =====
a_texts = []
with open(file_a, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.rstrip("\n").split("|")
        if len(parts) >= 2:
            a_texts.append(parts[2].strip())
        else:
            a_texts.append(None)

# ===== BUILD TEXT -> INDEX MAP FOR B =====
b_map = {}
with open(file_b, "r", encoding="utf-8") as f:
    for idx, line in enumerate(f):
        parts = line.rstrip("\n").split("|")
        if len(parts) >= 2:
            text = parts[2].strip()
            # store first occurrence
            if text not in b_map:
                b_map[text] = idx

# ===== PROCESS FILE C =====
found = 0
missing = 0

with open(file_c, "r", encoding="utf-8") as fc, \
     open(output_file, "w", encoding="utf-8") as out:

    for line in fc:
        # import pdb; pdb.set_trace()
        parts = line.rstrip("\n").split("|")
        if len(parts) < 2:
            continue

        c_col1 = parts[0].strip()
        c_text = parts[1].strip()

        if c_text in b_map:
            idx = b_map[c_text]

            if idx < len(a_texts) and a_texts[idx] is not None:
                a_text = a_texts[idx]
                out.write(f"{c_col1}|{c_text}|{a_text}\n")
                found += 1
            else:
                out.write(f"{c_col1}|{c_text}|NOT_FOUND_IN_A\n")
                missing += 1
        else:
            out.write(f"{c_col1}|{c_text}|NOT_FOUND_IN_B\n")
            missing += 1

print(f"Done. Found: {found}, Missing: {missing}")