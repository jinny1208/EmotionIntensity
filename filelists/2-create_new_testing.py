file_a = "/home/mila/j/jeony/EmotionIntensity/filelists/test_VCTK_onMEAD_test.txt"
file_b = "/home/mila/j/jeony/EmotionIntensity/filelists/vctk_test.txt.cleaned.redone"
file_c = "/home/mila/j/jeony/EmotionIntensity/filelists/difference.txt"

out_a = "/home/mila/j/jeony/EmotionIntensity/filelists/test_VCTK_onVCTKtextandMEADspk.txt"
out_b = "/home/mila/j/jeony/EmotionIntensity/filelists/test_VCTK_onVCTKtextandspk.txt"


with open(file_a, "r", encoding="utf-8") as fa, \
     open(file_b, "r", encoding="utf-8") as fb, \
     open(file_c, "r", encoding="utf-8") as fc, \
     open(out_a, "w", encoding="utf-8") as outA, \
     open(out_b, "w", encoding="utf-8") as outB:

    for i, line_c in enumerate(fc):
        line_a = fa.readline()
        line_b = fb.readline()

        # Stop if A or B runs out
        if not line_a or not line_b:
            print(f"Stopped early at line {i} (A or B exhausted)")
            break

        parts_a = line_a.strip().split("|")
        parts_b = line_b.strip().split("|")
        parts_c = line_c.strip().split("|")

        if len(parts_c) < 2:
            print(f"Skipping line {i}: malformed C")
            continue

        path_a = parts_a[0].strip()
        path_b = parts_b[0].strip()
        target = parts_c[1].strip()

        outA.write(f"{path_a}|{target}\n")
        outB.write(f"{path_b}|{target}\n")

print("Done.")