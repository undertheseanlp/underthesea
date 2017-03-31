
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
----------------------------

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

.. py:function:: underthesea.word_sent(sentence)

    segmented sentence

    :param unicode|str sentence: raw sentence
    :return: segmented sentence
    :rtype: unicode|str


Example for word segmentation:
    >>> from underthesea import word_sent
    >>> sentence ="Chúng ta thường nói đến Rau sạch , Rau an toàn để phân biệt với các rau bình thường bán ngoài chợ ."
    >>> sent = word_sent(sentence)
    >>> sent
    "Chúng_ta thường nói đến Rau_sạch , Rau an_toàn để phân_biệt với các rau bình_thường bán ngoài chợ ."



