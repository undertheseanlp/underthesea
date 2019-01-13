# -*- coding: utf-8 -*-
from unittest import TestCase
from underthesea.word_tokenize.regex_tokenize import tokenize


class TestWord_sent(TestCase):
    def test_tokenize(self):
        text = u"""Tổng thống Nga coi việc Mỹ không kích căn cứ quân sự của Syria là "sự gây hấn nhằm vào một quốc gia có chủ quyền", gây tổn hại đến quan hệ Moscow-Washington."""
        actual = tokenize(text, format="text")
        expected = u'Tổng thống Nga coi việc Mỹ không kích căn cứ quân sự của Syria là " sự gây hấn nhằm vào một quốc gia có chủ quyền " , gây tổn hại đến quan hệ Moscow - Washington .'
        self.assertEqual(actual, expected)

    def test_tokenize_2(self):
        text = u"""Vào tháng 10 năm ngoái, nghi phạm này từng bị bắt khi ăn cắp 25 mỹ phẩm trị giá 24.000 yen (hơn 200 USD) tại một cửa hàng giảm giá ở tỉnh Hyogo. Cùng năm, 4 thành viên khác cũng bị bắt vì ăn cắp một số lượng lớn son dưỡng môi và mỹ phẩm tại các cửa hiệu ở Osaka. Sau khi bị truy tố, 3 người trong số này được hưởng án treo."""
        actual = tokenize(text, format="text")
        expected = u'Vào tháng 10 năm ngoái , nghi phạm này từng bị bắt khi ăn cắp 25 mỹ phẩm trị giá 24.000 yen ( hơn 200 USD ) tại một cửa hàng giảm giá ở tỉnh Hyogo . Cùng năm , 4 thành viên khác cũng bị bắt vì ăn cắp một số lượng lớn son dưỡng môi và mỹ phẩm tại các cửa hiệu ở Osaka . Sau khi bị truy tố , 3 người trong số này được hưởng án treo .'
        self.assertEqual(actual, expected)

    def test_tokenize_3(self):
        text = u"""Giá khuyến mãi: 140.000đ / kg  ==> giảm được 20%"""
        actual = tokenize(text, format="text")
        expected = u'Giá khuyến mãi : 140.000 đ / kg ==> giảm được 20 %'
        self.assertEqual(actual, expected)

    def test_tokenize_4(self):
        text = u"""Microsoft hôm nay đã công bố cấu hình chính thức của thế hệ Xbox tiếp theo, tên mã Scorpio, và tin tưởng đây là loại console có sức mạnh xử lý cao nhất hiện giờ, mạnh hơn cả PS4 Pro. Tại buổi demo sản phẩm, Xbox Scorpio có thể chơi game đua xe Forza độ phân giải 4K với tốc độ đạt tới 60 khung hình/giây. Đồng thời Xbox Scorpio cũng sẽ tương thích với các tựa game Xbox cũ và quan trọng hơn nữa đó là các tựa game đó sẽ có đồ họa đẹp hơn, FPS cao hơn, thời gian load nhanh hơn khi chạy trên Xbox Scorpio."""
        actual = tokenize(text, format="text")
        expected = u'Microsoft hôm nay đã công bố cấu hình chính thức của thế hệ Xbox tiếp theo , tên mã Scorpio , và tin tưởng đây là loại console có sức mạnh xử lý cao nhất hiện giờ , mạnh hơn cả PS4 Pro . Tại buổi demo sản phẩm , Xbox Scorpio có thể chơi game đua xe Forza độ phân giải 4K với tốc độ đạt tới 60 khung hình / giây . Đồng thời Xbox Scorpio cũng sẽ tương thích với các tựa game Xbox cũ và quan trọng hơn nữa đó là các tựa game đó sẽ có đồ họa đẹp hơn , FPS cao hơn , thời gian load nhanh hơn khi chạy trên Xbox Scorpio .'
        self.assertEqual(actual, expected)

    def test_tokenize_5(self):
        text = u"""Tuyên bố của Điện Kremlin cũng nhấn mạnh, vụ tấn công sẽ “hủy hoại nghiêm trọng quan hệ Nga-Mỹ” và tạo ra “trở ngại cực lớn” cho việc tạo lập một liên minh quốc tế chống Tổ chức Nhà nước Hồi giáo tự xưng (IS)."""
        actual = tokenize(text, format="text")
        expected = u'Tuyên bố của Điện Kremlin cũng nhấn mạnh , vụ tấn công sẽ “ hủy hoại nghiêm trọng quan hệ Nga - Mỹ ” và tạo ra “ trở ngại cực lớn ” cho việc tạo lập một liên minh quốc tế chống Tổ chức Nhà nước Hồi giáo tự xưng ( IS ) .'
        self.assertEqual(actual, expected)

    def test_tokenize_6(self):
        text = u"""Ngày 11 tháng 3 là ngày thứ 70 (71 trong năm nhuận) trong lịch Gregory. Còn 295 ngày trong năm."""
        actual = tokenize(text, format="text")
        expected = u'Ngày 11 tháng 3 là ngày thứ 70 ( 71 trong năm nhuận ) trong lịch Gregory . Còn 295 ngày trong năm .'
        self.assertEqual(actual, expected)

    def test_tokenize_7(self):
        text = u"""Kết quả xổ số điện toán Vietlott ngày 6/2/2017"""
        actual = tokenize(text, format="text")
        expected = u"Kết quả xổ số điện toán Vietlott ngày 6/2/2017"
        self.assertEqual(actual, expected)

    def test_tokenize_8(self):
        text = u"""Theo thông báo kết luận thanh tra của UBND tỉnh Thanh Hoá sáng nay 30/3, giai đoạn 2010-2015 Sở Xây dựng Thanh Hoá đã bổ nhiệm một số trưởng phòng, phó phòng chưa có trình độ Trung cấp lý luận chính trị, chưa qua lớp bồi dưỡng nghiệp vụ quản lý nhà nước, không đúng quy định của UBND tỉnh Thanh Hoá.
"""
        actual = tokenize(text, format="text")
        expected = u"Theo thông báo kết luận thanh tra của UBND tỉnh Thanh Hoá sáng nay 30/3 , giai đoạn 2010 - 2015 Sở Xây dựng Thanh Hoá đã bổ nhiệm một số trưởng phòng , phó phòng chưa có trình độ Trung cấp lý luận chính trị , chưa qua lớp bồi dưỡng nghiệp vụ quản lý nhà nước , không đúng quy định của UBND tỉnh Thanh Hoá ."
        self.assertEqual(actual, expected)

    def test_datetime_1(self):
        text = u"""Ngày 6/2 6/2/2014 6-2 6-2-99 6.2 7.3.2014 2010-2015"""
        actual = tokenize(text, format="text")
        expected = u"Ngày 6/2 6/2/2014 6-2 6-2-99 6.2 7.3.2014 2010 - 2015"
        self.assertEqual(actual, expected)

    def test_abbreviations_1(self):
        text = u"""UBND. HĐND. TP."""
        actual = tokenize(text, format="text")
        expected = u"UBND. HĐND. TP."
        self.assertEqual(actual, expected)

    def test_urls(self):
        urls = [
            "google.com",
            "https://www.facebook.com/photo.php?fbid=1627680357512432&set=a.1406713109609159.1073741826.100008114498358&type=1"
        ]
        for url in urls:
            actual = tokenize(url, format="text")
            expected = url
            self.assertEqual(actual, expected)

    def test_urls_2(self):
        text = "việt đẹp mã.ftp://mp3.zing.vn vinh@zing.com.vn"
        actual = tokenize(text, format="text")
        expected = "việt đẹp mã . ftp://mp3.zing.vn vinh@zing.com.vn"
        self.assertEqual(actual, expected)
