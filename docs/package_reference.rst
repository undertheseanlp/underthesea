
word_sent
=========

.. automodule:: underthesea.word_sent

.. autofunction:: underthesea.word_sent.word_sent


pos_tag
=======

.. automodule:: underthesea.pos_tag

.. autofunction:: underthesea.pos_tag.pos_tag


chunking
========

.. automodule:: underthesea.chunking

.. autofunction:: underthesea.chunking.chunk


ner
===

.. automodule:: underthesea.ner

.. autofunction:: underthesea.ner.ner



classify
========

Make sure dependencies is installed and pretrained model is downloaded

.. code-block:: none

    $ pip install Cython
    $ pip install future scipy numpy scikit-learn
    $ pip install -U fasttext --no-cache-dir --no-deps --force-reinstall
    $ underthesea data

.. autofunction:: underthesea.classification.classify


sentiment
=========

.. autofunction:: underthesea.sentiment.sentiment
