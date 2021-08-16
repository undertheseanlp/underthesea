from data import Dictionary, Word

dict = Dictionary()
words = [
    Word('a', []),
    Word('b', [
        {
            'tag': 'Noun',
            'defs': [
                {
                    'def': 'abcdf',
                    'examples': ['1', '2']
                }
            ]
        }
    ])
]
for word in words:
    dict.add(word)
dict.save('test_dict.yaml')
