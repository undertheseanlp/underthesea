========================================
Under The Sea - Vietnamese NLP Toolkit
========================================


.. image:: https://img.shields.io/pypi/v/underthesea.svg
        :target: https://pypi.python.org/pypi/underthesea

.. image:: https://img.shields.io/travis/magizbox/underthesea.svg
        :target: https://travis-ci.org/magizbox/underthesea

.. image:: https://readthedocs.com/projects/magizbox-underthesea/badge/?version=latest
        :target: https://magizbox-underthesea.readthedocs-hosted.com/en/latest/?badge=latest
        :alt: Documentation Status
.. image:: https://pyup.io/repos/github/magizbox/underthesea/shield.svg
        :target: https://pyup.io/repos/github/magizbox/underthesea/
        :alt: Updates
|
.. image:: https://raw.githubusercontent.com/magizbox/underthesea/master/logo.jpg
        :target: https://raw.githubusercontent.com/magizbox/underthesea/master/logo.jpg

**underthesea** is a suite of open source Python modules, data sets and tutorials supporting research and development in Vietnamese Natural Language Processing.

* Free software: GNU General Public License v3
* Documentation: `https://underthesea.readthedocs.io <https://magizbox-underthesea.readthedocs-hosted.com/en/latest/>`_


Installation
----------------------------------------

To install underthesea, simply:

.. code-block:: bash

    $ pip install underthesea
    ✨🍰✨

Satisfaction, guaranteed.

Usage
----------------------------------------

* `1. Corpus <#1-corpus>`_
* `2. Word Segmentation <#2-word-segmentation>`_
* `3. POS Tagging <#3-pos-tagging>`_

****************************************
1. Corpus
****************************************

.. image:: https://img.shields.io/badge/documents-18k-red.svg

.. image:: https://img.shields.io/badge/words-74k-red.svg

Collection of Vietnamese corpus

* `Vietnamese Dictionary (74k words) <https://github.com/magizbox/underthesea/tree/master/underthesea/corpus/data>`_

* `Vietnamese News Corpus (10k documents) <https://github.com/magizbox/corpus.vinews>`_
* `Vietnamese Wikipedia Corpus (8k documents) <https://github.com/magizbox/corpus.viwiki>`_

****************************************
2. Word Segmentation
****************************************

.. image:: https://img.shields.io/badge/F1-97%25-red.svg

Vietnamese Word Segmentation using conditional random fields

* `Word Segmentation API <https://magizbox-underthesea.readthedocs-hosted.com/en/latest/api.html#word-sent-package>`_
* `Word Segmentation Experiences <https://github.com/magizbox/underthesea.word_sent>`_

.. code-block:: python

    from underthesea import word_sent
    sentence = u"Chúng ta thường nói đến Rau sạch , Rau an toàn để phân biệt với các rau bình thường bán ngoài chợ ."
    word_sent(sentence)
    # [u"Chúng ta", u"thường", u"nói", u"đến", u"Rau sạch", u",", u"Rau", u"an toàn", u"để", u"phân biệt", u"với",
    # u"các", u"rau", u"bình thường", u"bán", u"ngoài", u"chợ", u"."]
    word_sent(sentence, format="text")
    # u'Chúng_ta thường nói đến Rau_sạch , Rau an_toàn để phân_biệt với các rau bình_thường bán ngoài chợ .'

****************************************
3. POS Tagging
****************************************

Vietnamese Part of Speeding Tagging using Conditional Random Fields

* POS Tagging API
* `Pos Tagging Experiences <https://github.com/magizbox/underthesea.pos_tag>`_

Up Coming Features
----------------------------------------

* POS Tagging (API, )
* Word Representation (`Word Representation Experiences <https://github.com/magizbox/underthesea.word_representation>`_)
* Chunking (Experiences)
* Dependency Parsing (Experiences)
* Named Entity Recognition
* Sentiment Analysis
