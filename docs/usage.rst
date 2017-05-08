=====
Usage
=====

To use Under The Sea in a project::

    import underthesea

For word segmentation task::

    # -*- coding: utf-8 -*-
    from underthesea import word_sent
    sentence = u"Theo thông báo kết luận thanh tra của UBND tỉnh Thanh Hoá sáng nay 30/3, giai đoạn 2010-2015 Sở Xây dựng Thanh Hoá đã bổ nhiệm một số trưởng phòng, phó phòng chưa có trình độ Trung cấp lý luận chính trị, chưa qua lớp bồi dưỡng nghiệp vụ quản lý nhà nước, không đúng quy định của UBND tỉnh Thanh Hoá."
    word_sent(sentence)
    # [u"Theo", u"thông báo", u"kết luận", u"thanh tra", u"của", u"UBND", u"tỉnh", u"Thanh Hoá", u"sáng", u"nay", u"30", u"/", u"3", u",", u"giai đoạn", u"2010", u"-", u"2015", u"Sở Xây dựng", u"Thanh Hoá", u"đã", u"bổ nhiệm", u"một số", u"trưởng phòng", u",", u"phó phòng", u"chưa", u"có", u"trình độ", u"Trung cấp", u"lý luận", u"chính trị", u",", u"chưa", u"qua", u"lớp", u"bồi dưỡng", u"nghiệp vụ", u"quản lý", u"nhà nước", u",", u"không", u"đúng", u"quy định", u"của", u"UBND", u"tỉnh", u"Thanh Hoá", u"."][u"Theo", u"thông báo", u"kết luận", u"thanh tra", u"của", u"UBND", u"tỉnh", u"Thanh Hoá", u"sáng", u"nay", u"30", u"/", u"3", u",", u"giai đoạn", u"2010", u"-", u"2015", u"Sở Xây dựng", u"Thanh Hoá", u"đã", u"bổ nhiệm", u"một số", u"trưởng phòng", u",", u"phó phòng", u"chưa", u"có", u"trình độ", u"Trung cấp", u"lý luận", u"chính trị", u",", u"chưa", u"qua", u"lớp", u"bồi dưỡng", u"nghiệp vụ", u"quản lý", u"nhà nước", u",", u"không", u"đúng", u"quy định", u"của", u"UBND", u"tỉnh", u"Thanh Hoá", u"."]
