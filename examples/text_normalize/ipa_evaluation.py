from os.path import join

with open(join("outputs", "syllables_ipa.txt")) as f:
    lines1 = f.readlines()

with open(join("outputs", "vphon_syllables_ipa.txt")) as f:
    lines2 = f.readlines()

print(len(lines1))
print(len(lines2))
