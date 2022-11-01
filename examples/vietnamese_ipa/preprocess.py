def extract_sino():
    with open("tmp/sino_vietnamese_v1_2022.csv") as f:
        lines = f.read().splitlines()[1:]
    syllables = set()
    for line in lines:
        items = line.split(",")
        syllable = items[1].lower()
        syllables.add(syllable)
    syllables = sorted(list(syllables))
    content = "\n".join(syllables)
    with open("tmp/sino_vietnamese_syllables_v1_2022.txt", "w") as f:
        f.write(content)


if __name__ == '__main__':
    extract_sino()
