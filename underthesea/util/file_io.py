import io


def read(filename):
    with io.open(filename, "r", encoding="utf-8") as f:
        text = f.read()
    return text


def write(filename, text):
    with io.open(filename, "w", encoding="utf-8") as f:
        f.write(text)
