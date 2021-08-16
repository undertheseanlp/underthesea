from os.path import join, dirname
import joblib
import yaml

# SOURCE VERSION: v1.0-alpha
# TARGET VERSION: v1.0-alpha.1
# LAST UPDATE   : 2021/02/18

SCRIPTS_FOLDER = dirname(__file__)
DICT_FOLDER = join(dirname(SCRIPTS_FOLDER), 'corpus')
tmp_in_file = join(DICT_FOLDER, 'tmp_data.bin')
yaml_out_file = join(DICT_FOLDER, 'data_correct.yaml')
bin_out_file = join(DICT_FOLDER, 'tmp_data_correct.bin')

data = joblib.load(tmp_in_file)


def tag_correct(data):
    correct_data = data
    TAG_MAP = {
        'N': 'noun',
        'n': 'noun',  # special case "đạo đức"
        'A': 'adjective',
        'V': 'verb',
        'P': 'pronoun',
        'M': 'numeral',
        'D': 'determiner',
        'C': 'conjunction',
        'O': 'interjection',
        'R': 'adverb',
        'I': 'auxiliary',
        'E': 'preposition'
    }
    for word in data:
        for i, node in enumerate(data[word]):
            tag = node['tag']
            if tag in TAG_MAP:
                correct_data[word][i]['tag'] = TAG_MAP[tag]
            # case for word 'khối'
            if word == 'khối' and node['tag'] == 'S':
                correct_data[word][i]['tag'] = 'adjective'
    return correct_data


correct_data = tag_correct(data)
with open(yaml_out_file, 'w') as f:
    yaml.dump(correct_data, f, allow_unicode=True)
joblib.dump(correct_data, bin_out_file)
