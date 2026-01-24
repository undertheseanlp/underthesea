

def read(filename):
    with open(filename, encoding="utf-8") as f:
        text = f.read()
    return text


def write(filename, text):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(text)
