===============
Usage
===============

To use underthesea in your project::

    import underthesea

Word Segmentation
-------------------------

.. code-block:: python

    # -*- coding: utf-8 -*-
    >>> from underthesea import word_sent
    >>> sentence = u"Chúng ta thường nói đến Rau sạch , Rau an toàn để phân biệt với các rau bình thường bán ngoài chợ."

    >>> word_sent(sentence)
    [u"Chúng ta", u"thường", u"nói", u"đến", u"Rau sạch", u",", u"Rau", u"an toàn", u"để", u"phân biệt", u"với",
    u"các", u"rau", u"bình thường", u"bán", u"ngoài", u"chợ", u"."]

    >>> word_sent(sentence, format="text")
    u'Chúng_ta thường nói đến Rau_sạch , Rau an_toàn để phân_biệt với các rau bình_thường bán ngoài chợ .'

POS Tagging
-------------------------

.. code-block:: python

    # -*- coding: utf-8 -*-
    >>> from underthesea import pos_tag
    >>> text = u"Chợ thịt chó nổi tiếng ở TP Hồ Chí Minh bị truy quét"
    >>> pos_tag(text)
    [(u'Chợ', 'N'),
     (u'thịt', 'N'),
     (u'chó', 'N'),
     (u'nổi tiếng', 'A'),
     (u'ở', 'E'),
     (u'TP HCM', 'Np'),
     (u'bị', 'V'),
     (u'truy quét', 'V')]

Chunking
-------------------------

.. code-block:: python

    >>> # -*- coding: utf-8 -*-
    >>> from underthesea import chunk
    >>> text = u"Bác sĩ bây giờ có thể thản nhiên báo tin bệnh nhân bị ung thư?"
    >>> chunk(text)
    [(u'Bác sĩ', 'N', 'B-NP'),
     (u'bây giờ', 'P', 'I-NP'),
     (u'có thể', 'R', 'B-VP'),
     (u'thản nhiên', 'V', 'I-VP'),
     (u'báo tin', 'N', 'B-NP'),
     (u'bệnh nhân', 'N', 'I-NP'),
     (u'bị', 'V', 'B-VP'),
     (u'ung thư', 'N', 'I-VP'),
     (u'?', 'CH', 'O')]

Named Entity Recognition
-------------------------

.. code-block:: python

    >>> # -*- coding: utf-8 -*-
    >>> from underthesea import ner
    >>> text = u"Chưa tiết lộ lịch trình tới Việt Nam của Tổng thống Mỹ Donald Trump"
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

Text Classification
-------------------------

.. code-block:: python

    >>> # -*- coding: utf-8 -*-
    >>> from underthesea import classify
    >>> classify("HLV đầu tiên ở Premier League bị sa thải sau 4 vòng đấu")
    ['The thao']
    >>> classify("Hội đồng tư vấn kinh doanh Asean vinh danh giải thưởng quốc tế")
    ['Kinh doanh']
    >>> classify("Đánh giá “rạp hát tại gia” Samsung Soundbar Sound+ MS750")
    ['Vi tinh']

