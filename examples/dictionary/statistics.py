from os.path import realpath, dirname, join
import yaml

FOLDER = dirname(realpath(__file__))
content = yaml.safe_load(open(join(FOLDER, "new.yaml")))
num_words = len(content)
num_definitions = 0
num_examples = 0
for word in content:
    num_definitions += len(content[word])
    num_examples += sum([len(item['examples']) if 'examples' in item else 0 for item in content[word]])
total_data_points = num_words + num_definitions + num_examples

print(f'Total data points: {total_data_points}')

with open('stats.txt', 'w') as f:
    content = ''
    content += f'      Words: {num_words}\n'
    content += f'Definitions: {num_definitions}\n'
    content += f'   Examples: {num_examples}\n'
    content += f'Data points: {total_data_points}\n'
    f.write(content)