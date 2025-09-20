from collections import Counter

with open("tmp/sino-vietnamese-readings.csv") as f:
    lines = f.readlines()
    james_lines = [item.strip().split(",")[1].lower() for item in lines[1:]]
james_syllables = set(james_lines)
james_counter = Counter(james_lines)
with open("outputs/syllables.txt") as f:
    lines = f.readlines()
    underthesea_lines = [item.strip() for item in lines]

report = ""
report += "James Syllables\n"
report += "- Total    : " + str(len(james_lines)) + "\n"
report += "- Unique   : " + str(len(james_syllables)) + "\n"
james_duplicate = ", ".join([item[0] + ":" + str(item[1]) for item in james_counter.most_common(100)])
report += "- Duplicate: " + james_duplicate + "\n"
report += "\n"
report += "Underthesea Syllables\n"
report += "- Total : " + str(len(underthesea_lines)) + "\n"
report += "- Unique: " + str(len(set(underthesea_lines))) + "\n"

report += "\n"
underthesea_i_james = set(underthesea_lines).intersection(james_lines)
report += "Underthesea ∩ James\n"
report += "Intersection: " + str(len(underthesea_i_james)) + "\n"
report += "Sample      : " + ", ".join(list(underthesea_i_james)[:200]) + "\n"

report += "\n"
underthesea_james = set(underthesea_lines) - set(james_lines)
report += "Underthesea > James\n"
report += "Intersection: " + str(len(underthesea_james)) + "\n"
report += "Sample      : " + ", ".join(list(underthesea_james)[:200]) + "\n"

report += "\n"
james_underthesea = set(james_lines) - set(underthesea_lines)
report += "James > Underthesea" + "\n"
report += "Differences: " + str(len(james_underthesea)) + "\n"
report += "Sample     : " + ", ".join(list(james_underthesea)[:200]) + "\n"

print(report)

with open("outputs/underthesea_james_syllables_comparision.txt", "w") as f:
    f.write(report)
