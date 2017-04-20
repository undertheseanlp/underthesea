# -*- coding: utf-8 -*-
import io
import re
from collections import Counter
import pandas as pd


class TaggedCorpus:
    def __init__(self, documents=[]):
        self.documents = documents

    def save(self, filepath, format="UD", comment=True):
        with io.open(filepath, "w", encoding="utf8") as f:
            content = []
            for document in self.documents:
                if comment:
                    content.append(u"# document_id = %s\n" % document.id)
                for i, sentence in enumerate(document.sentences):
                    try:
                        if comment:
                            content.append(u"# sentence_id = %s\n" % sentence.id)
                    except:
                        if comment:
                            content.append(u"# sentence_id = %s-s%s\n" % (document.id, i + 1))
                    if comment:
                        content.append(u"# text = %s\n" % sentence.get_content())
                    for iw, words in enumerate(sentence.words):
                        content.append(u"%d\t%s\t%s\n" % (iw + 1, words.word, words.tags))
                    content.append(u"\n")
                content.append(u"\n")
            content = content[:-2]
            f.write(u"".join(content))

    def load(self, filepath, format="UD"):
        with open(filepath) as f:
            documents = []
            sentences = []
            words = []
            document_id = ""
            for line in f:
                matched_document = re.match("# document_id = (.*)\n", line)
                if matched_document:
                    document = TaggedDocument()
                    document.id = document_id
                    document.sentences = sentences
                    sentences = []
                    documents.append(document)
                    document_id = matched_document.group(1)
                    continue
                matched_sentence = re.match("# sentence_id = (.*)\n", line)
                if matched_sentence:
                    sentence_id = matched_sentence.group(1)
                    continue
                matched_text = re.match("# text = (.*)", line)
                if matched_text:
                    continue
                if line == "\n":
                    sentence = TaggedSentence()
                    sentence.id = sentence_id
                    sentence.words = words
                    words = []
                    sentences.append(sentence)
                    continue
                i, word, tag, x = re.split("\n|\t", line)
                word = TaggedWord(word.decode("utf-8"), tag)
                words.append(word)
            document = TaggedDocument()
            document.id = document_id
            sentence = TaggedSentence()
            sentence.id = sentence_id
            sentence.words = words
            sentences.append(sentence)
            document.sentences = sentences
            documents.append(document)
            documents = documents[1:]
            self.documents = documents

    def sents(self):
        sentences = [sent for document in self.documents for sent in document.sentences]
        sentences = [sent for sent in sentences if sent != []]
        return sentences

    def words(self):
        sents = self.sents()
        words = [word for sent in sents for word in sent.words]
        return words

    def analyze(self):
        tagged_words = self.words()
        data = dict()
        data["total_words"] = len(tagged_words)
        words = [tagged_word.word for tagged_word in tagged_words]
        tags = [tagged_word.tags for tagged_word in tagged_words]
        word_counter = Counter(words)
        tag_counter = Counter(tags)
        df = pd.DataFrame.from_dict(word_counter, orient='index').reset_index()
        df.to_excel("words.xlsx", encoding="utf-8", index=False)
        df = pd.DataFrame.from_dict(tag_counter, orient='index').reset_index()
        df.to_excel("tags.xlsx", encoding="utf-8", index=False)
        return data


class TaggedDocument:
    def __init__(self, sentences=[]):
        self.sentences = sentences


class TaggedSentence:
    def __init__(self, tagged_words=[]):
        self.words = tagged_words

    def get_content(self):
        words = [tagged_word.word for tagged_word in self.words]
        content = u" ".join(words)
        for punctuation in ["...", ",", ")", "\""]:
            content = content.replace(" " + punctuation, punctuation)
        for punctuation in ["(", "\""]:
            content = content.replace(punctuation + " ", punctuation)
        return content


class TaggedWord:
    def __init__(self, word, tags):
        self.word = word
        self.tags = tags
