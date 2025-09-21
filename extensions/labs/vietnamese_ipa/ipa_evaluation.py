from os.path import join

with open(join("../text_normalize/outputs", "syllables_ipa.txt")) as f:
    lines1 = f.readlines()

with open(join("../text_normalize/outputs", "vphon_syllables_ipa.txt")) as f:
    lines2 = f.readlines()

print(len(lines1))
print(len(lines2))
