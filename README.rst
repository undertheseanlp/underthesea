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

+-----------------+---------------------------------------------------------------------------------------------------------------------+
| Free software   | GNU General Public License v3                                                                                       |
+-----------------+---------------------------------------------------------------------------------------------------------------------+
| Live demo       | `undertheseanlp.com <http://undertheseanlp.com/>`_                                                                  |
+-----------------+---------------------------------------------------------------------------------------------------------------------+
| Colab notebooks | stable: `v1.1.17 <https://colab.research.google.com/drive/1U6EWY7ewNUtCXGsa5uZtDEz4I5exO_fo>`_                      |
|                 | ,                                                                                                                   |
|                 | latest: `v1.2.0 <https://colab.research.google.com/drive/1gD8dSMSE_uNacW4qJ-NSnvRT85xo9ZY2>`_ |
+-----------------+---------------------------------------------------------------------------------------------------------------------+
| Documentation   | `Underthesea Documentation <http://underthesea.readthedocs.io/en/latest/>`_                                         |
+-----------------+---------------------------------------------------------------------------------------------------------------------+
| Facebook        | `Underthesea Page <https://www.facebook.com/undertheseanlp/>`_                                                      |
+-----------------+---------------------------------------------------------------------------------------------------------------------+
| Youtube         | `Underthesea NLP Channel <https://www.youtube.com/channel/UC9Jv1Qg49uprg6SjkyAqs9A>`_                               |
+-----------------+---------------------------------------------------------------------------------------------------------------------+

* Free software: GNU General Public License v3
* Live demo: `undertheseanlp.com <http://undertheseanlp.com/>`_
* Colab Notebooks: `stable: v1.1.17 <https://colab.research.google.com/drive/1U6EWY7ewNUtCXGsa5uZtDEz4I5exO_fo>`_, `latest: v1.2.0 <https://colab.research.google.com/drive/1gD8dSMSE_uNacW4qJ-NSnvRT85xo9ZY2#scrollTo=qFOO2WqcKgYr>`_
* Documentation: `https://underthesea.readthedocs.io <http://underthesea.readthedocs.io/en/latest/>`_
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
* `8. Vietnamese NLP Resources <#8-vietnamese-nlp-resources>`_

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
    >>> from underthesea import sent_tokenize
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

Download models

.. code-block:: bash

    $ underthesea download-model TC_GENERAL
    $ underthesea download-model TC_BANK

Usage

.. code-block:: python

    >>> # -*- coding: utf-8 -*-
    >>> from underthesea import classify

    >>> classify('HLV đầu tiên ở Premier League bị sa thải sau 4 vòng đấu')
    ['The thao']
    >>> classify('Hội đồng tư vấn kinh doanh Asean vinh danh giải thưởng quốc tế')
    ['Kinh doanh']

    >> classify('Lãi suất từ BIDV rất ưu đãi', domain='bank')
    ['INTEREST_RATE']

****************************************
7. Sentiment Analysis
****************************************

.. image:: https://img.shields.io/badge/F1-59.5%25-red.svg
		:target: https://github.com/undertheseanlp/sentiment

.. image:: https://img.shields.io/badge/✎-custom%20models-blue.svg
    :target: https://github.com/undertheseanlp/sentiment

Download models

.. code-block:: bash

    $ underthesea download-model SA_GENERAL
    $ underthesea download-model SA_BANK


Usage


.. code-block:: python

    >>> # -*- coding: utf-8 -*-
    >>> from underthesea import sentiment

    >>> sentiment('hàng kém chất lg,chăn đắp lên dính lông lá khắp người. thất vọng')
    negative
    >>> sentiment('Sản phẩm hơi nhỏ so với tưởng tượng nhưng chất lượng tốt, đóng gói cẩn thận.')
    positive

    >>> sentiment('Đky qua đường link ở bài viết này từ thứ 6 mà giờ chưa thấy ai lhe hết', domain='bank')
    ['CUSTOMER_SUPPORT#negative']
    >>> sentiment('Xem lại vẫn thấy xúc động và tự hào về BIDV của mình', domain='bank')
    ['TRADEMARK#positive']

****************************************
8. Vietnamese NLP Resources
****************************************

List resources

.. code-block:: bash

    $ underthesea list-data
    | Name         | Type        | License   |   Year | Directory             |
    |--------------+-------------+-----------+--------+-----------------------|
    | UTS2017-BANK | Categorized | Open      |   2017 | datasets/UTS2017-BANK |
    | VNESES       | Plaintext   | Open      |   2012 | datasets/LTA          |
    | VNTQ_BIG     | Plaintext   | Open      |   2012 | datasets/LTA          |
    | VNTQ_SMALL   | Plaintext   | Open      |   2012 | datasets/LTA          |
    | VNTC         | Categorized | Open      |   2007 | datasets/VNTC         |

    $ underthesea list-data --all

Download resources

.. code-block:: bash

    $ underthesea download-data VNTC
    100%|██████████| 74846806/74846806 [00:09<00:00, 8243779.16B/s]
    Resource VNTC is downloaded in ~/.underthesea/datasets/VNTC folder

Up Coming Features
----------------------------------------

* Text to Speech
* Automatic Speech Recognition
* Machine Translation
* Dependency Parsing

Contributing
----------------------------------------

Do you want to contribute with underthesea development? Great! Please read more details at `CONTRIBUTING.rst. <https://github.com/undertheseanlp/underthesea/blob/master/CONTRIBUTING.rst>`_
