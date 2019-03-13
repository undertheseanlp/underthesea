====================================
Underthesea - Vietnamese NLP Toolkit
====================================


.. image:: https://img.shields.io/pypi/v/underthesea.svg
        :target: https://pypi.python.org/pypi/underthesea

.. image:: https://img.shields.io/pypi/pyversions/underthesea.svg
        :target: https://pypi.python.org/pypi/underthesea

.. image:: https://img.shields.io/badge/license-GNU%20General%20Public%20License%20v3-brightgreen.svg
        :target: https://pypi.python.org/pypi/underthesea

.. image:: https://img.shields.io/travis/undertheseanlp/underthesea.svg
        :target: https://travis-ci.org/undertheseanlp/underthesea

.. image:: https://readthedocs.org/projects/underthesea/badge/?version=latest
        :target: http://underthesea.readthedocs.io/en/latest/
        :alt: Documentation Status

.. image:: https://img.shields.io/badge/chat-on%20facebook-green.svg
    :target: https://www.facebook.com/undertheseanlp/

|

.. image:: https://raw.githubusercontent.com/undertheseanlp/underthesea/master/logo.jpg
        :target: https://raw.githubusercontent.com/undertheseanlp/underthesea/master/logo.jpg

**underthesea** is a suite of open source Python modules, data sets and tutorials supporting research and development in Vietnamese Natural Language Processing.

* Free software: GNU General Public License v3
* Documentation: `https://underthesea.readthedocs.io <http://underthesea.readthedocs.io/en/latest/>`_
* Live demo: `undertheseanlp.com <http://undertheseanlp.com/>`_
* Facebook Page: `https://www.facebook.com/undertheseanlp/ <https://www.facebook.com/undertheseanlp/>`_
* Youtube: `Underthesea NLP Channel <https://www.youtube.com/channel/UC9Jv1Qg49uprg6SjkyAqs9A>`_

Installation
----------------------------------------

To install underthesea, simply:

.. code-block:: bash

    $ pip install underthesea
    ✨🍰✨

Satisfaction, guaranteed.

Usage
----------------------------------------

* `1. Sentence Segmentation <#1-sentence-segmentation>`_
* `2. Word Segmentation <#2-word-segmentation>`_
* `3. POS Tagging <#3-pos-tagging>`_
* `4. Chunking <#4-chunking>`_
* `5. Named Entity Recognition <#5-named-entity-recognition>`_
* `6. Text Classification <#6-text-classification>`_
* `7. Sentiment Analysis <#7-sentiment-analysis>`_

****************************************
1. Sentence Segmentation
****************************************

.. image:: https://img.shields.io/badge/F1-98%25-red.svg
        :target: https://github.com/undertheseanlp/sent_tokenize

.. image:: https://img.shields.io/badge/✎-custom%20models-blue.svg
        :target: https://github.com/undertheseanlp/sent_tokenize

Usage

.. code-block:: python

    >>> # -*- coding: utf-8 -*-
    >>> from underthesea import word_tokenize
    >>> text = 'Taylor cho biết lúc đầu cô cảm thấy ngại với cô bạn thân Amanda nhưng rồi mọi thứ trôi qua nhanh chóng. Amanda cũng thoải mái với mối quan hệ này.'

    >>> sent_tokenize(text)
    [
        "Taylor cho biết lúc đầu cô cảm thấy ngại với cô bạn thân Amanda nhưng rồi mọi thứ trôi qua nhanh chóng.",
        "Amanda cũng thoải mái với mối quan hệ này."
    ]

****************************************
2. Word Segmentation
****************************************

.. image:: https://img.shields.io/badge/F1-94%25-red.svg
        :target: https://github.com/undertheseanlp/word_tokenize

.. image:: https://img.shields.io/badge/✎-custom%20models-blue.svg
        :target: https://github.com/undertheseanlp/word_tokenize

Usage

.. code-block:: python

    >>> # -*- coding: utf-8 -*-
    >>> from underthesea import word_tokenize
    >>> sentence = 'Chàng trai 9X Quảng Trị khởi nghiệp từ nấm sò'

    >>> word_tokenize(sentence)
    ['Chàng trai', '9X', 'Quảng Trị', 'khởi nghiệp', 'từ', 'nấm', 'sò']

    >>> word_tokenize(sentence, format="text")
    'Chàng_trai 9X Quảng_Trị khởi_nghiệp từ nấm sò'

****************************************
3. POS Tagging
****************************************

.. image:: https://img.shields.io/badge/accuracy-92.3%25-red.svg
        :target: https://github.com/undertheseanlp/pos_tag

.. image:: https://img.shields.io/badge/✎-custom%20models-blue.svg
        :target: https://github.com/undertheseanlp/pos_tag

Usage

.. code-block:: python

    >>> # -*- coding: utf-8 -*-
    >>> from underthesea import pos_tag
    >>> pos_tag('Chợ thịt chó nổi tiếng ở Sài Gòn bị truy quét')
    [('Chợ', 'N'),
     ('thịt', 'N'),
     ('chó', 'N'),
     ('nổi tiếng', 'A'),
     ('ở', 'E'),
     ('Sài Gòn', 'Np'),
     ('bị', 'V'),
     ('truy quét', 'V')]

****************************************
4. Chunking
****************************************

.. image:: https://img.shields.io/badge/F1-77%25-red.svg
		:target: https://github.com/undertheseanlp/chunking

.. image:: https://img.shields.io/badge/✎-custom%20models-blue.svg
		:target: https://github.com/undertheseanlp/chunking

Usage

.. code-block:: python

    >>> # -*- coding: utf-8 -*-
    >>> from underthesea import chunk
    >>> text = 'Bác sĩ bây giờ có thể thản nhiên báo tin bệnh nhân bị ung thư?'
    >>> chunk(text)
    [('Bác sĩ', 'N', 'B-NP'),
     ('bây giờ', 'P', 'I-NP'),
     ('có thể', 'R', 'B-VP'),
     ('thản nhiên', 'V', 'I-VP'),
     ('báo tin', 'N', 'B-NP'),
     ('bệnh nhân', 'N', 'I-NP'),
     ('bị', 'V', 'B-VP'),
     ('ung thư', 'N', 'I-VP'),
     ('?', 'CH', 'O')]

****************************************
5. Named Entity Recognition
****************************************

.. image:: https://img.shields.io/badge/F1-86.6%25-red.svg
		:target: https://github.com/undertheseanlp/ner

.. image:: https://img.shields.io/badge/✎-custom%20models-blue.svg
		:target: https://github.com/undertheseanlp/ner

Usage

.. code-block:: python

    >>> # -*- coding: utf-8 -*-
    >>> from underthesea import ner
    >>> text = 'Chưa tiết lộ lịch trình tới Việt Nam của Tổng thống Mỹ Donald Trump'
    >>> ner(text)
    [('Chưa', 'R', 'O', 'O'),
     ('tiết lộ', 'V', 'B-VP', 'O'),
     ('lịch trình', 'V', 'B-VP', 'O'),
     ('tới', 'E', 'B-PP', 'O'),
     ('Việt Nam', 'Np', 'B-NP', 'B-LOC'),
     ('của', 'E', 'B-PP', 'O'),
     ('Tổng thống', 'N', 'B-NP', 'O'),
     ('Mỹ', 'Np', 'B-NP', 'B-LOC'),
     ('Donald', 'Np', 'B-NP', 'B-PER'),
     ('Trump', 'Np', 'B-NP', 'I-PER')]

****************************************
6. Text Classification
****************************************

.. image:: https://img.shields.io/badge/accuracy-86.7%25-red.svg
    :target: https://github.com/undertheseanlp/classification

.. image:: https://img.shields.io/badge/✎-custom%20models-blue.svg
    :target: https://github.com/undertheseanlp/classification

Install dependencies and download default model

.. code-block:: bash

    $ pip install Cython
    $ pip install joblib future scipy numpy scikit-learn
    $ pip install -U fasttext --no-cache-dir --no-deps --force-reinstall
    $ underthesea data

Usage

.. code-block:: python

    >>> # -*- coding: utf-8 -*-
    >>> from underthesea import classify
    >>> classify('HLV đầu tiên ở Premier League bị sa thải sau 4 vòng đấu')
    ['The thao']
    >>> classify('Hội đồng tư vấn kinh doanh Asean vinh danh giải thưởng quốc tế')
    ['Kinh doanh']
    >>> classify('Đánh giá “rạp hát tại gia” Samsung Soundbar Sound+ MS750')
    ['Vi tinh']

****************************************
7. Sentiment Analysis
****************************************

.. image:: https://img.shields.io/badge/F1-59.5%25-red.svg
		:target: https://github.com/undertheseanlp/sentiment

.. image:: https://img.shields.io/badge/✎-custom%20models-blue.svg
    :target: https://github.com/undertheseanlp/sentiment

Install dependencies

.. code-block:: bash

    $ pip install future scipy numpy scikit-learn==0.19.2 joblib

Usage


.. code-block:: python

    >>> # -*- coding: utf-8 -*-
    >>> from underthesea import sentiment
    >>> sentiment('Gọi mấy lần mà lúc nào cũng là các chuyên viên đang bận hết ạ', domain='bank')
    ('CUSTOMER SUPPORT#NEGATIVE',)
    >>> sentiment('bidv cho vay hay ko phu thuoc y thich cua thang tham dinh, ko co quy dinh ro rang', domain='bank')
    ('LOAN#NEGATIVE',)

Up Coming Features
----------------------------------------

* Text to Speech
* Automatic Speech Recognition
* Machine Translation
* Dependency Parsing

Contributing
----------------------------------------

Do you want to contribute with underthesea development? Great! Please read more details at `CONTRIBUTING.rst. <https://github.com/undertheseanlp/underthesea/blob/master/CONTRIBUTING.rst>`_
