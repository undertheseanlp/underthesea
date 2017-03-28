def word2features(sent, i):
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

    if i >= 2:
        word2 = sent[i - 2][0]
        features.extend(['-2:word.lower=' + word2.lower()])
    else:
        features.append('BOS')

    if i < len(sent) - 1:
        next_word_1 = sent[i+1][0]
        features.extend([
            '+1:word.lower=' + next_word_1.lower()
        ])
    else:
        features.append('EOS')

    if i < len(sent) - 2:
        next_word_2 = sent[i+2][0]
        features.extend([
            '+2:word.lower=' + next_word_2.lower()
        ])
    else:
        features.append('EOS')

    return features
