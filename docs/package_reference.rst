=============
word_tokenize
=============

.. autofunction:: underthesea.word_tokenize.word_tokenize

=======
pos_tag
=======

.. autofunction:: underthesea.pos_tag.pos_tag

========
chunking
========

.. autofunction:: underthesea.chunking.chunk

===
ner
===

.. autofunction:: underthesea.ner.ner

========
classify
========

Install dependencies and download default model

.. code-block:: bash

    $ pip install Cython
    $ pip install future scipy numpy scikit-learn
    $ pip install -U fasttext --no-cache-dir --no-deps --force-reinstall
    $ underthesea data

.. autofunction:: underthesea.classification.classify

=========
sentiment
=========

Install dependencies

.. code-block:: bash

    $ pip install future scipy numpy scikit-learn==0.19.2 joblib

.. autofunction:: underthesea.sentiment.sentiment
