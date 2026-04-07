filelist1 = "/home/mila/j/jeony/EmotionIntensity/filelists/test_VCTK_onMEAD_test.txt"
filelist2 = "/home/mila/j/jeony/EmotionIntensity/filelists/ljs_test_val.txt"
output_file = "/home/mila/j/jeony/EmotionIntensity/filelists/test_VCTK_onMEADandLJS.txt"

with open(filelist1, "r", encoding="utf-8") as f1, \
     open(filelist2, "r", encoding="utf-8") as f2, \
     open(output_file, "w", encoding="utf-8") as fout:

    for line1, line2 in zip(f1, f2):
        parts1 = line1.strip().split("|")
        parts2 = line2.strip().split("|")

        if len(parts1) < 1 or len(parts2) < 2:
            continue

        col1 = parts1[0]   # first element from filelist1
        col2 = parts2[1]   # second element from filelist2

        fout.write(f"{col1}|{col2}\n")