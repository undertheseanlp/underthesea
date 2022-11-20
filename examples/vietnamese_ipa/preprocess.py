def extract_sino():
    with open("tmp/sino_vietnamese_v1_2022.csv") as f:
        lines = f.read().splitlines()[1:]
    syllables = set()
    ignore_syllables = ["tê (tây)", "uái (khoá"]
    for line in lines:
        items = line.split(",")
        syllable = items[1].lower()
        syllables.add(syllable)
        if syllable not in ignore_syllables:
            syllables.add(syllable)
    syllables = sorted(list(syllables))
    content = "\n".join(syllables)
    with open("tmp/sino_vietnamese_syllables_v1_2022.txt", "w") as f:
        f.write(content)


def preprocess_viwik():
    with open("tmp/viwik_syllables_raw_v1_2022.txt") as f:
        lines = f.read().splitlines()
    syllables = set()
    ignore_syllables = ["gĩữ", "by"]
    for line in lines:
        if line not in ignore_syllables:
            syllables.add(line)
    syllables = sorted(syllables)
    content = "\n".join(syllables)
    with open("tmp/viwik_syllables_v1_2022.txt", "w") as f:
        f.write(content)


def generate_uts_syllables():
    with open("tmp/sino_vietnamese_syllables_v1_2022.txt") as f:
        sino_syllables = set(f.read().strip().splitlines())

    with open("tmp/viwik_syllables_v1_2022.txt") as f:
        viwik_syllables = set(f.read().strip().splitlines())

    syllables = sino_syllables.union(viwik_syllables)
    syllables = sorted(syllables)
    content = "\n".join(syllables)
    with open("tmp/uts_syllables_v1_2022.txt", "w") as f:
        f.write(content)


if __name__ == '__main__':
    extract_sino()
    preprocess_viwik()
    generate_uts_syllables()
