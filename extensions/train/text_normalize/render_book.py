def generate_ipa_tables():
    # output = "2 & cell2 & cell3 \\\\"
    with open("outputs/syllables_ipa.txt") as f:
        content = f.read()
    lines = content.strip().split("\n")
    output = ""
    # i = 0
    for line in lines:
        # i += 1
        # if i > 200:
        #     break
        items = line.split(",")
        order, syllable, ipa = items
        ipa = "/" + ipa + "/"
        output += f"{order}. {syllable} {ipa} \\\\\n"
    return output


if __name__ == '__main__':
    with open("templates/book.tex") as f:
        template = f.read()

    content = template
    ipa_latex = generate_ipa_tables()
    content = content.replace("% IPA CONTENT", ipa_latex)

    with open("outputs/books/book.tex", "w") as f:
        f.write(content)
