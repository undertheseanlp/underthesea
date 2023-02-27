from os.path import join, dirname
import os
import shutil
import data

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


templates_dir = join(pwd, "tmp", "statistics_template")


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

tags = {}
for sentence in corpus.train:
    for row in sentence:
        word, pos = row
        if pos in tags:
            tags[pos] += 1
        else:
            tags[pos] = 1

# create list pos with format: "A (1)<br/> B(2)"
list_pos = " - ".join([f"<a href='{key}.html'>{key} ({value}</a>)" for key, value in tags.items()])
dataset_name = "VLSP2013 POS Tagging"
data = {
    "DATASET_NAME": dataset_name,
    "LIST_POS": list_pos
} 
render("index.html", "index.html", data)

for tag in tags:
    tag_data = {
        "DATASET_NAME": dataset_name,
        "TAG_NAME": tag,
        "MOST_FREQUENT_WORDS": "a b c"
    }
    render("tag.html", f"{tag}.html", tag_data)
