from os.path import join
import pandas as pd
from underthesea.utils.vietnamese_ipa import Syllable, vietnamese_sort_key


def generate_ipa_syllables(filename):
    with open(filename) as f:
        items = f.readlines()
    items = [item.strip() for item in items]
    items = sorted(items, key=vietnamese_sort_key)
    data = []
    for item in items:
        text = item
        try:
            syllable = Syllable(text)
        except Exception:
            continue
        ipa = syllable.generate_ipa()
        components = syllable.matched.groupdict()
        C1 = components['C1']

        row = {
            "syllable": text,
            "ipa": ipa,
            "C1": C1,
            "w": components['w'],
            "V": components['V'],
            "C2": components['C2'],
        }
        data.append(row)
    df = pd.DataFrame(data)

    # write Excel file
    df.index = df.index + 1
    df = df.reset_index()
    df.to_excel(join("tmp", "syllables_ipa.xlsx"), index=False)

    # write text file
    result = ""
    for index, row in df.iterrows():
        i = row['index']
        text = row['syllable']
        ipa = row['ipa']
        result += f"{i},{text},{ipa}\n"
    with open(join("tmp", "syllables_ipa.txt"), "w") as f:
        f.write(result)


if __name__ == '__main__':
    filename = join("tmp", "uts_syllables_v1_2022.txt")
    generate_ipa_syllables(filename)
