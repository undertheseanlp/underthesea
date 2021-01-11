# -*- coding: utf-8 -*-
import pickle
import string
from os.path import join, dirname

from nltk import PunktSentenceTokenizer

sentence_tokenizer = None


def _load_model():
    global sentence_tokenizer
    if sentence_tokenizer is not None:
        return
    model_path = join(dirname(__file__), 'st_kiss-strunk-2006_2019_01_13.pkl')
    with open(model_path, 'rb') as fs:
        punkt_param = pickle.load(fs)

    punkt_param.sent_starters = {}
    abbrev_types = ['g.m.t', 'e.g', 'dr', 'dr', 'vs', "000", 'mr', 'mrs', 'prof', 'inc', 'tp', 'ts', 'ths',
                    'th', 'vs', 'tp', 'k.l', 'a.w.a.k.e', 't', 'a.i', '</i', 'g.w',
                    'ass',
                    'u.n.c.l.e', 't.e.s.t', 'ths', 'd.c', 've…', 'ts', 'f.t', 'b.b', 'z.e', 's.g', 'm.p',
                    'g.u.y',
                    'l.c', 'g.i', 'j.f', 'r.r', 'v.i', 'm.h', 'a.s', 'bs', 'c.k', 'aug', 't.d.q', 'b…', 'ph',
                    'j.k', 'e.l', 'o.t', 's.a']
    abbrev_types.extend(string.ascii_uppercase)
    for abbrev_type in abbrev_types:
        punkt_param.abbrev_types.add(abbrev_type)
    for abbrev_type in string.ascii_lowercase:
        punkt_param.abbrev_types.add(abbrev_type)
    sentence_tokenizer = PunktSentenceTokenizer(punkt_param)


def sent_tokenize(text):
    global sent_tokenizer
    _load_model()
    sentences = sentence_tokenizer.sentences_from_text(text)
    return sentences
