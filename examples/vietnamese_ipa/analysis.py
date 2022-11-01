def compare_hieuthi_viwik():
    with open("tmp/hieuthi_all_vietnamese_syllables_v1_2022.txt") as f:
        syllables = f.read().strip().splitlines()
        ht_syllables = set(syllables)

    with open("tmp/viwik_syllables_v1_2022.txt") as f:
        syllables = f.read().strip().splitlines()
        vw_syllables = set(syllables)

    content = ""
    content += f"ht_syllables: {len(ht_syllables)}\n"
    content += f"vw_syllables: {len(vw_syllables)}\n"
    ht_vw_diff = sorted(list(ht_syllables - vw_syllables))
    vw_ht_diff = sorted(list(vw_syllables - ht_syllables))
    content += f"ht - vw: {len(ht_vw_diff)}\n"
    content += f"vw - ht: {len(vw_ht_diff)}\n"
    print(content)

    with open("tmp/report_compare_hieuthi_viwik", "w") as f:
        f.write(content)

    with open("tmp/report_ht_vw_diff.txt", "w") as f:
        content = "\n".join(ht_vw_diff)
        f.write(content)

    with open("tmp/report_vw_ht_diff.txt", "w") as f:
        content = "\n".join(vw_ht_diff)
        f.write(content)


if __name__ == '__main__':
    compare_hieuthi_viwik()
