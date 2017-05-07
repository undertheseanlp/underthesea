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

.. py:data:: underthesea.corpus.viet_dict_74K

    Usage

    >>> from underthesea.corpur import viet_dict_74K
    >>> viet_dict_74K.words()
    ["a", "A", "a-ba-giua", ...]

.. automodule:: underthesea.corpus.corpus
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

.. py:function:: underthesea.word_sent(sentence,text = False)

    segmented sentence

    :param bool: text
    :param unicode|str sentence: raw sentence
    :return: segmented sentence
    :rtype: unicode|str


    Usage:
            >>> from underthesea import word_sent
            >>> sentence ="Chúng ta thường nói đến Rau sạch , Rau an toàn để phân biệt với các rau bình thường bán ngoài chợ ."
            >>> word_sent(sentence)
            "Chúng_ta thường nói đến Rau_sạch , Rau an_toàn để phân_biệt với các rau bình_thường bán ngoài chợ ."

            >>> word_sent(sentence, text=True)
            [u'Theo', u'thông báo', u'kết luận', u'thanh tra', u'của', u'UBND',
                    u'tỉnh', u'Thanh Hoá', u'sáng', u'nay', u'30', u'/', u'3', u',', u'giai đoạn',
                    u'2010', u'-', u'2015', u'Sở Xây dựng', u'Thanh Hoá', u'đã',
                    u'bổ nhiệm', u'một số', u'trưởng phòng', u',', u'phó phòng',
                    u'chưa', u'có', u'trình độ', u'Trung cấp', u'lý luận',
                    u'chính trị', u',', u'chưa', u'qua', u'lớp', u'bồi dưỡng',
                    u'nghiệp vụ', u'quản lý', u'nhà nước', u',', u'không',
                    u'đúng', u'quy định', u'của', u'UBND', u'tỉnh', u'Thanh Hoá', u'.']


.. automodule:: underthesea.word_sent.tokenize
    :members:
    :undoc-members:
    :show-inheritance:
    Usage:
            >>> from underthesea.word_sent.tokenize import tokenize
            >>> text = "Đám cháy bùng phát trưa nay, 7/4, tại khu nhà tôn ngay gần tòa nhà Keangnam, đường Phạm Hùng. Ngọn lửa cùng khói đen bốc lên dữ dội làm đen kịt một góc không gian. Giao thông quanh khu vực bị ảnh hưởng, trong đó đường trên cao bị tắc một đoạn khá dài..."
            >>> tokenize(text)
            "Đám cháy bùng phát trưa nay , 7 / 4 , tại khu nhà tôn ngay gần tòa nhà Keangnam , đường Phạm Hùng . Ngọn lửa cùng khói đen bốc lên dữ dội làm đen kịt một góc không gian . Giao thông quanh khu vực bị ảnh hưởng , trong đó đường trên cao bị tắc một đoạn khá dài ..."

:mod:`pos_tag` Package
-------------------------


.. py:function:: underthesea.pos_tag(sentence, text = False)

    pos tagged sentence

    :param bool: text
    :param unicode|str sentence: raw sentence
    :return: pos tagged sentence
    :rtype: unicode|str

    Usage:
                >>> from underthesea import word_sent
                >>> sentence ='Theo dự kiến, lễ bốc thăm vòng bán kết Champions League sẽ diễn ra vào lúc 18h00 ngày thứ Sáu (21/4) tại Nyon, Thụy Sĩ. Cũng như vòng tứ kết, vòng đấu này không có đội hạt giống, các CLB cùng quốc gia phải đụng độ với nhau.'
                >>> pos_tag(sentence)
                'Theo/E dự kiến/N ,/CH lễ/N bốc/V thăm/V vòng/N bán kết/N Champions League/Np sẽ/R diễn/V ra/R vào/V lúc/N 18h00/V ngày/N thứ Sáu/N (/CH 21/M //CH 4/M )/CH tại/E Nyon/Np ,/CH Thụy Sĩ/Np ./CH Cũng/Np như/C vòng/N tứ kết/N ,/CH vòng/N đấu/V này/P không/R có/V đội/N hạt giống/N ,/CH các/L CLB/Ny cùng/A quốc gia/N phải/V đụng độ/V với/E nhau/N ./CH'

                >>> from underthesea import word_sent
                >>> sentence ='Chủ nhiệm UB Đối ngoại của Quốc hội Nguyễn Văn Giàu phân tích, ở nước ngoài, ông từng chứng kiến việc doanh nghiệp tặng xe, phương tiện cho tổ chức, nhưng ở Việt Nam, việc này dường như có nhiều… lắt léo.'
                >>> pos_tag(sentence, True)
                [(1, u'Chủ nhiệm', 'Nc'), (2, u'UB', 'Np'), (3, u'Đối ngoại', 'Np'),
                        (4, u'của', 'E'), (5, u'Quốc hội', 'Np'), (6, u'Nguyễn Văn Giàu', 'Np'),
                        (7, u'phân tích', 'V'), (8, u',', 'CH'), (9, u'ở', 'E'),
                        (10, u'nước ngoài', 'N'), (11, u',', 'CH'), (12, u'ông', 'N'),
                        (13, u'từng', 'R'), (14, u'chứng kiến', 'V'), (15, u'việc', 'N'),
                        (16, u'doanh nghiệp', 'V'), (17, u'tặng', 'V'), (18, u'xe', 'N'), (19, u',', 'CH'),
                        (20, u'phương tiện', 'V'), (21, u'cho', 'E'), (22, u'tổ chức', 'N'),
                        (23, u',', 'CH'), (24, u'nhưng', 'C'), (25, u'ở', 'E'), (26, u'Việt Nam', 'Np'),
                        (27, u',', 'CH'), (28, u'việc', 'N'), (29, u'này', 'P'),
                        (30, u'dường như', 'N'), (31, u'có', 'V'), (32, u'nhiều', 'A'),
                        (33, u'…', 'CH'), (34, u'lắt léo', 'V'), (35, u'.', 'CH')]

