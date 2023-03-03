from os.path import join, dirname
import os
import shutil
import data
import pandas as pd

working_folder = dirname(__file__)
dataset_folder = join(working_folder, "vlsp2013")

corpus = data.DataReader.load_tagged_corpus(
    join(working_folder, "tmp/vlsp2013"), train_file="train.txt", test_file="test.txt"
)
pwd = dirname(__file__)

statistics_folder = join(pwd, "tmp/vlsp2013_pos_statistics")
if os.path.exists(statistics_folder):
    shutil.rmtree(statistics_folder)
os.makedirs(statistics_folder)


templates_dir = join(pwd, "statistics_template")


def render(infile, outfile, variables={}):
    # read file from  template folder
    with open(join(templates_dir, infile), "r") as f:
        template = f.read()
    # replace variables
    content = template
    for key, value in variables.items():
        template_key = f"<!-- VARIABLE:{key} -->"
        content = content.replace(template_key, value)
    # write to statistics folder
    with open(join(statistics_folder, outfile), "w") as f:
        f.write(content)


# create dataframe with 3 columns: word, pos, freq
df = pd.DataFrame(columns=["pos", "word", "freq"])

words_tags = {}
for sentence in corpus.train:
    for row in sentence:
        row = tuple(row)
        word, pos = row
        if row not in words_tags:
            words_tags[row] = 1
        else:
            words_tags[row] += 1

# convert words_tags to dataframe
words_tags_data = []
for row in words_tags:
    word, pos = row
    freq = words_tags[row]
    words_tags_data.append({"pos": pos, "word": word, "freq": freq})
df = pd.DataFrame(words_tags_data)

df.to_excel(join(pwd, "tmp/words_tags_freq.xlsx"), index=False)
# create list pos with format: "A (1)<br/> B(2)"
# list_pos = " - ".join([f"<a href='{key}.html'>{key} ({value}</a>)" for key, value in tags.items()])
# dataset_name = "VLSP2013 POS Tagging"
# data = {
#     "DATASET_NAME": dataset_name,
#     "LIST_POS": list_pos
# }
# render("index.html", "index.html", data)

# for tag in tags:
#     num_freq = 200
#     most_frequent_words_data = sorted(words_tags[tag].items(), key=lambda x: x[1], reverse=True)[:num_freq]
#     most_frequent_words = ", ".join([f"<i>{key}</i> ({value})" for key, value in most_frequent_words_data])
#     tag_data = {
#         "DATASET_NAME": dataset_name,
#         "TAG_NAME": tag,
#         "MOST_FREQUENT_WORDS": most_frequent_words,
#         "NUM_FREQ": str(num_freq)
#     }
#     render("tag.html", f"{tag}.html", tag_data)
