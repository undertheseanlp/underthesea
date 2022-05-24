from os.path import realpath, dirname, join
import yaml

FOLDER = dirname(realpath(__file__))
content = yaml.safe_load(open(join(FOLDER, "new.yaml")))
num_words = len(content)
num_definitions = 0
for word in content:
    num_definitions += len(content[word])


with open('stats.txt', 'w') as f:
    content = ''
    content += f'      Words: {num_words}\n'
    content += f'Definitions: {num_definitions}\n'
    f.write(content)