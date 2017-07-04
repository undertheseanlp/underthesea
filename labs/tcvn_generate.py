import pandas as pd


def generate_character(c):
    return unichr(int("0x" + c, 16))


s = "\u" + "103"
print(u'\u0103')
df = pd.read_excel("tcvn_6909_2001.xlsx", index=False)
df["c"] = df.apply(lambda row: generate_character(row["unicode"]), axis=1)

df.to_excel("tcvn_6909_2001_generate.xlsx", index=False)
print(0)
