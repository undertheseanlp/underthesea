
def compare_file(file_1, name_1, file_2, name_2):
    with open(file_1) as f:
        syllables = f.read().strip().splitlines()
        syllables_1 = set(syllables)

    with open(file_2) as f:
        syllables = f.read().strip().splitlines()
        syllables_2 = set(syllables)

    content = f"Compare {name_1} - {name_2}\n"
    content += f"{name_1}: {len(syllables_1)}\n"
    content += f"{name_2}: {len(syllables_2)}\n"
    union_1_2 = sorted(list(syllables_1.union(syllables_2)))
    inter_1_2 = sorted(list(syllables_1.intersection(syllables_2)))
    diff_1_2 = sorted(list(syllables_1 - syllables_2))
    diff_2_1 = sorted(list(syllables_2 - syllables_1))
    content += f"{name_1} union {name_2}: {len(union_1_2)}\n"
    content += f"{name_1} inter {name_2}: {len(inter_1_2)}\n"
    content += f"{name_1} - {name_2}: {len(diff_1_2)}\n"
    content += f"{name_2} - {name_1}: {len(diff_2_1)}\n"
    print(content)

    with open(f"tmp/report_compare_{name_1}_{name_2}.txt", "w") as f:
        f.write(content)

    with open(f"tmp/report_{name_1}_{name_2}_union.txt", "w") as f:
        content = "\n".join(union_1_2)
        f.write(content)

    with open(f"tmp/report_{name_1}_{name_2}_inter.txt", "w") as f:
        content = "\n".join(inter_1_2)
        f.write(content)

    with open(f"tmp/report_{name_1}_{name_2}_diff.txt", "w") as f:
        content = "\n".join(diff_1_2)
        f.write(content)

    with open(f"tmp/report_{name_2}_{name_1}_diff.txt", "w") as f:
        content = "\n".join(diff_2_1)
        f.write(content)


if __name__ == '__main__':
    # compare hieuthi and viwik
    file_1 = "tmp/hieuthi_all_vietnamese_syllables_v1_2022.txt"
    name_1 = "ht"
    file_2 = "tmp/viwik_syllables_v1_2022.txt"
    name_2 = "vw"
    compare_file(file_1, name_1, file_2, name_2)

    # compare sino and viwik
    file_1 = "tmp/sino_vietnamese_syllables_v1_2022.txt"
    name_1 = "sino"
    file_2 = "tmp/viwik_syllables_v1_2022.txt"
    name_2 = "vw"
    compare_file(file_1, name_1, file_2, name_2)
