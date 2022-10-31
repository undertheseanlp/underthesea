.. _api:

Developer Interface
===================

.. module:: underthesea

=============
word_tokenize
=============

.. autofunction:: word_tokenize

=======
pos_tag
=======

.. autofunction:: pos_tag

========
chunking
========

.. autofunction:: chunk

===
ner
===

.. autofunction:: ner

========
classify
========

Install dependencies and download default model

.. code-block:: bash

    $ pip install Cython
    $ pip install future scipy numpy scikit-learn
    $ pip install -U fasttext --no-cache-dir --no-deps --force-reinstall
    $ underthesea data

.. autofunction:: classify

=========
sentiment
=========

Install dependencies

.. code-block:: bash

    $ pip install future scipy numpy scikit-learn==0.19.2 joblib

.. autofunction:: sentiment

========
viet2ipa
========

.. autofunction:: underthesea.pipeline.ipa.viet2ipa
