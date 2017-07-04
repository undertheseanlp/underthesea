API
====

:mod:`underthesea` Package
----------------------------

.. automodule:: underthesea
    :members:
    :undoc-members:
    :inherited-members:
    :show-inheritance:

.. automodule:: underthesea.underthesea
    :members:
    :undoc-members:
    :inherited-members:
    :show-inheritance:

:mod:`corpus` Package
--------------------------

.. automodule:: underthesea.corpus
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: underthesea.corpus.corpus
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: underthesea.corpus.corpus.readers
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: underthesea.corpus.readers.dictionary_loader
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: underthesea.corpus.plaintext
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: underthesea.corpus.document
    :members:
    :undoc-members:
    :show-inheritance:


:mod:`transformer` Package
----------------------------------------

.. automodule:: underthesea.transformer.unicode
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: underthesea.transformer.lowercase
    :members:
    :undoc-members:
    :show-inheritance:


:mod:`word_sent` Package
-------------------------

.. py:function:: underthesea.word_sent.tokenize(sentence)

    tokenize a sentence

    :param text: raw text input
    :return: tokenize text
    :rtype: unicode|str

.. code-block:: python

	# -*- coding: utf-8 -*-
	>>> from underthesea.word_sent.tokenize import tokenize
	>>> text = u"Đám cháy bùng phát trưa nay, 7/4, tại khu nhà tôn ngay gần tòa nhà Keangnam, đường Phạm Hùng. Ngọn lửa cùng khói đen bốc lên dữ dội làm đen kịt một góc không gian. Giao thông quanh khu vực bị ảnh hưởng, trong đó đường trên cao bị tắc một đoạn khá dài..."

	>>> tokenize(text)
	u"Đám cháy bùng phát trưa nay , 7 / 4 , tại khu nhà tôn ngay gần tòa nhà Keangnam , đường Phạm Hùng . Ngọn lửa cùng khói đen bốc lên dữ dội làm đen kịt một góc không gian . Giao thông quanh khu vực bị ảnh hưởng , trong đó đường trên cao bị tắc một đoạn khá dài ..."

.. py:function:: underthesea.word_sent(sentence)

    word segmentation

    :param unicode|str sentence: raw sentence
    :return: segmented sentence
    :rtype: unicode|str

.. code-block:: python

    >>> # -*- coding: utf-8 -*-
    >>> from underthesea import word_sent
    >>> sentence = u"Chúng ta thường nói đến Rau sạch , Rau an toàn để phân biệt với các rau bình thường bán ngoài chợ ."

    >>> word_sent(sentence)
    [u"Chúng ta", u"thường", u"nói", u"đến", u"Rau sạch", u",", u"Rau", u"an toàn", u"để", u"phân biệt", u"với",
    u"các", u"rau", u"bình thường", u"bán", u"ngoài", u"chợ", u"."]

    >>> word_sent(sentence, format="text")
    u'Chúng_ta thường nói đến Rau_sạch , Rau an_toàn để phân_biệt với các rau bình_thường bán ngoài chợ .'

:mod:`pos_tag` Package
-------------------------

.. py:function:: underthesea.pos_tag(sentence)

    part of speech tagging

    :param unicode|str sentence: raw sentence
    :return: tagged sentence
    :rtype: list

.. code-block:: python

    >>> # -*- coding: utf-8 -*-
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

:mod:`chunking` Package
-------------------------

.. py:function:: underthesea.chunk(sentence)

    chunk a sentence to phrases

    :param unicode sentence: raw sentence
    :return: list of tuple with word, pos tag, chunking tag
    :rtype: list

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
