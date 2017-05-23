import time

from underthesea.corpus.readers.dictionary_loader import DictionaryLoader

words = set(DictionaryLoader('Viet74K.txt').words)


def word2features(sent, i):
    """
    add feature for each word

    :param unicode|str sent: input sentence
    :param int i: index of word in sentence
    :return: word added feature
    :rtype: list
    """
    word = sent[i][0]
    features = [
        word,
        'bias',
        'word.lower=' + word.lower(),
        'word.isupper=%s' % word.isupper(),
        'word.istitle=%s' % word.istitle(),
        'word.isdigit=%s' % word.isdigit(),
    ]
    if i >= 1:
        word1 = sent[i - 1][0]
        features.extend([
            '-1:word.lower=' + word1.lower(),
            '-1:word.istitle=%s' % word1.istitle(),
            '-1:word.isupper=%s' % word1.isupper(),
        ])
    else:
        features.append('BOS')

    if i < len(sent) - 1:
        word = " ".join([sent[i][0].lower(), sent[i + 1][0].lower()])
        features.extend([
            '2_word.in_dictionary=' + str(int(word in words)),
        ])
    else:
        features.append('2_word:EOS')

    if i >= 1:
        word = " ".join([sent[i - 1][0].lower(), sent[i][0].lower()])
        features.extend([
            '-2_word.in_dictionary=' + str(int(word in words)),
        ])
    else:
        features.append('-2_word:BOS')

    if i >= 2:
        word = " ".join([sent[i - 2][0].lower(), sent[i - 1][0].lower(), sent[i][0].lower()])
        features.extend([
            '-3_word.in_dictionary=' + str(int(word in words)),
        ])
    else:
        features.append('-3_word:BOS')

    if i < len(sent) - 2:
        word = " ".join([sent[i][0].lower(), sent[i + 1][0].lower(), sent[i + 2][0]])
        features.extend([
            '3_word.in_dictionary=' + str(int(word in words)),
        ])
    else:
        features.append('3_word:EOS')

    if i >= 2:
        word2 = sent[i - 2][0]
        features.extend(['-2:word.lower=' + word2.lower()])
    else:
        features.append('BOS')

    if i < len(sent) - 1:
        next_word_1 = sent[i + 1][0]
        features.extend([
            '+1:word.lower=' + next_word_1.lower()
        ])
    else:
        features.append('EOS')

    if i < len(sent) - 2:
        next_word_2 = sent[i + 2][0]
        features.extend([
            '+2:word.lower=' + next_word_2.lower()
        ])
    else:
        features.append('EOS')

    return features
