========================================================
Underthesea - Công cụ xử lý ngôn ngữ tự nhiên tiếng Việt
========================================================

.. image:: https://img.shields.io/pypi/v/underthesea.svg
        :target: https://pypi.python.org/pypi/underthesea

.. image:: https://img.shields.io/pypi/pyversions/underthesea.svg
        :target: https://pypi.python.org/pypi/underthesea

.. image:: https://img.shields.io/pypi/l/underthesea.svg
        :target: https://pypi.python.org/pypi/underthesea

.. image:: https://img.shields.io/travis/magizbox/underthesea.svg
        :target: https://travis-ci.org/magizbox/underthesea

.. image:: https://readthedocs.org/projects/underthesea/badge/?version=latest
        :target: http://underthesea.readthedocs.io/en/latest/
        :alt: Documentation Status

.. image:: https://pyup.io/repos/github/magizbox/underthesea/shield.svg
        :target: https://pyup.io/repos/github/magizbox/underthesea/
        :alt: Updates

.. image:: https://img.shields.io/badge/liên%20hệ-qua%20facebook-green.svg
    :target: https://www.facebook.com/undertheseanlp/

|

`[English] <https://github.com/magizbox/underthesea/>`_
`[Tiếng Việt] <https://github.com/magizbox/underthesea/blob/master/README.vi.rst>`_

.. image:: https://raw.githubusercontent.com/magizbox/underthesea/master/logo.jpg
        :target: https://raw.githubusercontent.com/magizbox/underthesea/master/logo.jpg

**underthesea** là tập hợp các dự án, nguồn dữ liệu mở, tài liệu hướng dẫn với mục đích hỗ trợ việc nghiên cứu và phát triển trong lĩnh vực xử lý ngôn ngữ tự nhiên tiếng Việt.

* Nguồn mở: GNU General Public License v3
* Tài liệu: `https://underthesea.readthedocs.io <http://underthesea.readthedocs.io/en/latest/>`_
* Demo: `underthesea app <http://magizbox.com:9386/#/>`_
* Facebook Page: `https://www.facebook.com/undertheseanlp/ <https://www.facebook.com/undertheseanlp/>`_

Cài đặt
----------------------------------------

Việc cài đặt underthesea được thực hiện thông qua pip

.. code-block:: bash

    $ pip install underthesea==1.1.8
    ✨🍰✨


Sử dụng
----------------------------------------

* `1. Tách từ <#1-tách-từ>`_
* `2. Gán nhãn từ loại <#2-gán-nhãn-từ-loại>`_
* `3. Xác định cụm từ <#3-xác-định-cụm-từ>`_
* `4. Nhận diện thực thể có tên <#4-nhận-diện-thực-thể-có-tên>`_
* `5. Phân loại văn bản <#5-phân-loại-văn-bản>`_
* `6. Phân tích cảm xúc <#6-phân-tích-cảm-xúc>`_


****************************************
1. Tách từ
****************************************

.. image:: https://img.shields.io/badge/F1-94%25-red.svg
        :target: https://github.com/magizbox/underthesea.word_sent

.. image:: https://img.shields.io/badge/✎-tùy%20biến%20mô%20hình-blue.svg
        :target: https://github.com/undertheseanlp/word_sent

.. image:: https://img.shields.io/badge/★-api-green.svg
    :target: http://underthesea.readthedocs.io/en/latest/package_reference.html#word_sent

Sử dụng

.. code-block:: python

    >>> # -*- coding: utf-8 -*-
    >>> from underthesea import word_tokenize
    >>> sentence = 'Chàng trai 9X Quảng Trị khởi nghiệp từ nấm sò'

    >>> word_tokenize(sentence)
    ['Chàng trai', '9X', 'Quảng Trị', 'khởi nghiệp', 'từ', 'nấm', 'sò']

    >>> word_tokenize(sentence, format="text")
    'Chàng_trai 9X Quảng_Trị khởi_nghiệp từ nấm sò'

****************************************
2. Gán nhãn từ loại
****************************************

.. image:: https://img.shields.io/badge/accuracy-92.3%25-red.svg
        :target: https://github.com/magizbox/underthesea.pos_tag

.. image:: https://img.shields.io/badge/✎-tùy%20biến%20mô%20hình-blue.svg
        :target: https://github.com/undertheseanlp/pos_tag

.. image:: https://img.shields.io/badge/★-api-green.svg
    :target: http://underthesea.readthedocs.io/en/latest/package_reference.html#pos-tag

Sử dụng

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
3. Xác định cụm từ
****************************************

.. image:: https://img.shields.io/badge/F1-77%25-red.svg
		:target: https://github.com/magizbox/underthesea.chunking

.. image:: https://img.shields.io/badge/✎-tùy%20biến%20mô%20hình-blue.svg
		:target: https://github.com/undertheseanlp/chunking

.. image:: https://img.shields.io/badge/★-api-green.svg
    :target: http://underthesea.readthedocs.io/en/latest/package_reference.html#chunking

Sử dụng

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
4. Nhận diện thực thể có tên
****************************************

.. image:: https://img.shields.io/badge/F1-86.6%25-red.svg
		:target: https://github.com/magizbox/underthesea.ner

.. image:: https://img.shields.io/badge/✎-tùy%20biến%20mô%20hình-blue.svg
		:target: https://github.com/undertheseanlp/ner

.. image:: https://img.shields.io/badge/★-api-green.svg
    :target: http://underthesea.readthedocs.io/en/latest/package_reference.html#ner

Sử dụng

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
5. Phân loại văn bản
****************************************

.. image:: https://img.shields.io/badge/accuracy-86.7%25-red.svg
    :target: https://github.com/magizbox/underthesea.classification

.. image:: https://img.shields.io/badge/✎-tùy%20biến%20mô%20hình-blue.svg
    :target: https://github.com/undertheseanlp/classification

.. image:: https://img.shields.io/badge/★-api-green.svg
    :target: http://underthesea.readthedocs.io/en/latest/package_reference.html#classify

Cài đặt các gói liên quan và tải mô hình

.. code-block:: bash

    $ pip install Cython
    $ pip install joblib future scipy numpy scikit-learn
    $ pip install -U fasttext --no-cache-dir --no-deps --force-reinstall
    $ underthesea data

Sử dụng

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
6. Phân tích cảm xúc
****************************************

.. image:: https://img.shields.io/badge/F1-59.5%25-red.svg
		:target: https://github.com/undertheseanlp/sentiment

.. image:: https://img.shields.io/badge/✎-tùy%20biến%20mô%20hình-blue.svg
    :target: https://github.com/undertheseanlp/sentiment

.. image:: https://img.shields.io/badge/★-api-green.svg
    :target: http://underthesea.readthedocs.io/en/latest/package_reference.html#sentiment

Cài đặt các gói liên quan

.. code-block:: bash

    $ pip install future scipy numpy scikit-learn==0.19.0 joblib

Sử dụng


.. code-block:: python

    >>> # -*- coding: utf-8 -*-
    >>> from underthesea import sentiment
    >>> sentiment('Gọi mấy lần mà lúc nào cũng là các chuyên viên đang bận hết ạ', domain='bank')
    ('CUSTOMER SUPPORT#NEGATIVE',)
    >>> sentiment('bidv cho vay hay ko phu thuoc y thich cua thang tham dinh, ko co quy dinh ro rang', domain='bank')
    ('LOAN#NEGATIVE',)

Các tính năng sắp tới
----------------------------------------

* Tổng hợp tiếng nói
* Nhận dạng tiếng nói
* Dịch máy
* Phân tích cú pháp phụ thuộc

Đóng góp
----------------------------------------

Bạn đang muốn đóng góp cho dự án underthesea? Tuyệt vời! Đọc hướng dẫn chi tiết tại `CONTRIBUTING.rst. <https://github.com/magizbox/underthesea/blob/master/CONTRIBUTING.rst>`_
