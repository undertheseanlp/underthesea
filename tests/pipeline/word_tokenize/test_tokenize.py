# -*- coding: utf-8 -*-
from unittest import TestCase
from underthesea.pipeline.word_tokenize.regex_tokenize import tokenize


class TestTokenize(TestCase):
    def test_tokenize(self):
        text = u"""Tổng thống Nga coi việc Mỹ không kích căn cứ quân sự của Syria là "sự gây hấn nhằm vào một quốc gia có chủ quyền", gây tổn hại đến quan hệ Moscow-Washington."""
        actual = tokenize(text, format="text")
        expected = u'Tổng thống Nga coi việc Mỹ không kích căn cứ quân sự của Syria là " sự gây hấn nhằm vào một quốc gia có chủ quyền " , gây tổn hại đến quan hệ Moscow-Washington .'
        self.assertEqual(expected, actual)

    def test_tokenize_2(self):
        text = u"""Vào tháng 10 năm ngoái, nghi phạm này từng bị bắt khi ăn cắp 25 mỹ phẩm trị giá 24.000 yen (hơn 200 USD) tại một cửa hàng giảm giá ở tỉnh Hyogo. Cùng năm, 4 thành viên khác cũng bị bắt vì ăn cắp một số lượng lớn son dưỡng môi và mỹ phẩm tại các cửa hiệu ở Osaka. Sau khi bị truy tố, 3 người trong số này được hưởng án treo."""
        actual = tokenize(text, format="text")
        expected = u'Vào tháng 10 năm ngoái , nghi phạm này từng bị bắt khi ăn cắp 25 mỹ phẩm trị giá 24.000 yen ( hơn 200 USD ) tại một cửa hàng giảm giá ở tỉnh Hyogo . Cùng năm , 4 thành viên khác cũng bị bắt vì ăn cắp một số lượng lớn son dưỡng môi và mỹ phẩm tại các cửa hiệu ở Osaka . Sau khi bị truy tố , 3 người trong số này được hưởng án treo .'
        self.assertEqual(expected, actual)

    def test_tokenize_3(self):
        text = u"""Giá khuyến mãi: 140.000đ / kg  ==> giảm được 20%"""
        actual = tokenize(text, format="text")
        expected = u'Giá khuyến mãi : 140.000 đ / kg ==> giảm được 20 %'
        self.assertEqual(expected, actual)

    def test_tokenize_4(self):
        text = u"""Microsoft hôm nay đã công bố cấu hình chính thức của thế hệ Xbox tiếp theo, tên mã Scorpio, và tin tưởng đây là loại console có sức mạnh xử lý cao nhất hiện giờ, mạnh hơn cả PS4 Pro. Tại buổi demo sản phẩm, Xbox Scorpio có thể chơi game đua xe Forza độ phân giải 4K với tốc độ đạt tới 60 khung hình/giây. Đồng thời Xbox Scorpio cũng sẽ tương thích với các tựa game Xbox cũ và quan trọng hơn nữa đó là các tựa game đó sẽ có đồ họa đẹp hơn, FPS cao hơn, thời gian load nhanh hơn khi chạy trên Xbox Scorpio."""
        actual = tokenize(text, format="text")
        expected = u'Microsoft hôm nay đã công bố cấu hình chính thức của thế hệ Xbox tiếp theo , tên mã Scorpio , và tin tưởng đây là loại console có sức mạnh xử lý cao nhất hiện giờ , mạnh hơn cả PS4 Pro . Tại buổi demo sản phẩm , Xbox Scorpio có thể chơi game đua xe Forza độ phân giải 4K với tốc độ đạt tới 60 khung hình / giây . Đồng thời Xbox Scorpio cũng sẽ tương thích với các tựa game Xbox cũ và quan trọng hơn nữa đó là các tựa game đó sẽ có đồ họa đẹp hơn , FPS cao hơn , thời gian load nhanh hơn khi chạy trên Xbox Scorpio .'
        self.assertEqual(actual, expected)

    def test_tokenize_5(self):
        text = u"""Tuyên bố của Điện Kremlin cũng nhấn mạnh, vụ tấn công sẽ “hủy hoại nghiêm trọng quan hệ Nga-Mỹ” và tạo ra “trở ngại cực lớn” cho việc tạo lập một liên minh quốc tế chống Tổ chức Nhà nước Hồi giáo tự xưng (IS)."""
        actual = tokenize(text, format="text")
        expected = u'Tuyên bố của Điện Kremlin cũng nhấn mạnh , vụ tấn công sẽ “ hủy hoại nghiêm trọng quan hệ Nga-Mỹ ” và tạo ra “ trở ngại cực lớn ” cho việc tạo lập một liên minh quốc tế chống Tổ chức Nhà nước Hồi giáo tự xưng ( IS ) .'
        self.assertEqual(expected, actual)

    def test_tokenize_6(self):
        text = u"""Ngày 11 tháng 3 là ngày thứ 70 (71 trong năm nhuận) trong lịch Gregory. Còn 295 ngày trong năm."""
        actual = tokenize(text, format="text")
        expected = u'Ngày 11 tháng 3 là ngày thứ 70 ( 71 trong năm nhuận ) trong lịch Gregory . Còn 295 ngày trong năm .'
        self.assertEqual(expected, actual)

    def test_tokenize_7(self):
        text = u"""Kết quả xổ số điện toán Vietlott ngày 6/2/2017"""
        actual = tokenize(text, format="text")
        expected = u"Kết quả xổ số điện toán Vietlott ngày 6/2/2017"
        self.assertEqual(expected, actual)

    def test_tokenize_8(self):
        text = u"""Theo thông báo kết luận thanh tra của UBND tỉnh Thanh Hoá sáng nay 30/3, giai đoạn 2010-2015 Sở Xây dựng Thanh Hoá đã bổ nhiệm một số trưởng phòng, phó phòng chưa có trình độ Trung cấp lý luận chính trị, chưa qua lớp bồi dưỡng nghiệp vụ quản lý nhà nước, không đúng quy định của UBND tỉnh Thanh Hoá.
"""
        actual = tokenize(text, format="text")
        expected = u"Theo thông báo kết luận thanh tra của UBND tỉnh Thanh Hoá sáng nay 30/3 , giai đoạn 2010 - 2015 Sở Xây dựng Thanh Hoá đã bổ nhiệm một số trưởng phòng , phó phòng chưa có trình độ Trung cấp lý luận chính trị , chưa qua lớp bồi dưỡng nghiệp vụ quản lý nhà nước , không đúng quy định của UBND tỉnh Thanh Hoá ."
        self.assertEqual(expected, actual)

    def test_special(self):
        text = 'v.v...'
        actual = tokenize(text, format="text")
        expected = 'v.v...'
        self.assertEqual(expected, actual)

    def test_abbreviation_1(self):
        text = u"""UBND. HĐND. TP."""
        actual = tokenize(text, format="text")
        expected = u"UBND. HĐND. TP ."
        self.assertEqual(expected, actual)

    def test_abbreviation_2(self):
        text = 'Toàn cảnh lễ ký kết giữa công ty Tân Thạnh A và công ty Lotte E&C'
        actual = tokenize(text, format="text")
        expected = 'Toàn cảnh lễ ký kết giữa công ty Tân Thạnh A và công ty Lotte E&C'
        self.assertEqual(expected, actual)

    def test_abbreviation_3(self):
        text = 'L.ANH N.ẨN Chị T.T.M.'
        actual = tokenize(text, format="text")
        expected = 'L.ANH N.ẨN Chị T.T.M.'
        self.assertEqual(expected, actual)

    def test_abbreviation_4(self):
        text = 'một ở làng H\'Lũ H\'Mông'
        actual = tokenize(text, format="text")
        expected = 'một ở làng H\'Lũ H\'Mông'
        self.assertEqual(expected, actual)

    def test_url(self):
        urls = [
            "google.com",
            "https://www.facebook.com/photo.php?fbid=1627680357512432&set=a.1406713109609159.1073741826.100008114498358&type=1"
        ]
        for url in urls:
            actual = tokenize(url, format="text")
            expected = url
            self.assertEqual(expected, actual)

    def test_url_2(self):
        text = "việt đẹp mã.ftp://mp3.zing.vn vinh@zing.com.vn"
        actual = tokenize(text, format="text")
        expected = "việt đẹp mã . ftp://mp3.zing.vn vinh@zing.com.vn"
        self.assertEqual(expected, actual)

    def test_datetime_1(self):
        text = "29-10-2004 Ngày 2/2014 6/2 6/2/2014 6-2 6-2-99 6.2 7.3.2014 2010-2015 2004/09/15 08:41:40"
        actual = tokenize(text, format="text")
        expected = "29-10-2004 Ngày 2/2014 6/2 6/2/2014 6-2 6-2-99 6.2 7.3.2014 2010 - 2015 2004/09/15 08:41:40"
        self.assertEqual(expected, actual)

    def test_datetime_2(self):
        text = "vào tháng 5-2002 , 11-2003 và 8-3-2004"
        actual = tokenize(text, format="text")
        expected = "vào tháng 5-2002 , 11-2003 và 8-3-2004"
        self.assertEqual(expected, actual)

    def test_name(self):
        text = "Xe gắn máy số 53S5 - 3720"
        actual = tokenize(text, format="text")
        expected = "Xe gắn máy số 53S5 - 3720"
        self.assertEqual(expected, actual)

    def test_number(self):
        text = "tổng cộng 60.542.000 đồng 100,000,000"
        actual = tokenize(text, format="text")
        expected = "tổng cộng 60.542.000 đồng 100,000,000"
        self.assertEqual(expected, actual)

    def test_number_2(self):
        text = "1.600m-2.000m Nó chỉ có mặt ở vùng đất độ cao 1.600m-2.000m so với mặt biển."
        actual = tokenize(text, format="text")
        expected = "1.600 m - 2.000 m Nó chỉ có mặt ở vùng đất độ cao 1.600 m - 2.000 m so với mặt biển ."
        self.assertEqual(expected, actual)

    def test_emoji(self):
        text = 'Tầm giá quá rẻ để mua 1 cục bực tức để trên người :)) <3'
        actual = tokenize(text, format="text")
        expected = 'Tầm giá quá rẻ để mua 1 cục bực tức để trên người :)) <3'
        self.assertEqual(expected, actual)

    def test_word_hyphen(self):
        text = 'Tiêm kích F-16 Bỉ kích bom Su-34 Nga Tàu Apolo-2 Đoàn tàu SE-4 kiểm soát 49X-6666 Rolls-Royce'
        actual = tokenize(text, format="text")
        expected = 'Tiêm kích F-16 Bỉ kích bom Su-34 Nga Tàu Apolo-2 Đoàn tàu SE-4 kiểm soát 49X-6666 Rolls-Royce'
        self.assertEqual(expected, actual)

    def test_punct(self):
        text = '(baodautu.vn) Trao đổi với báo chí chiều 27/5, ông Lê Quốc Bình, TGĐ Công ty cổ phần đầu tư hạ tầng kỹ thuật TP. HCM (CII) chính thức thông tin: UBND TP vừa có quyết định cho phép CII được thu phí hoàn vốn đầu tư dự án xây dựng cầu Rạch Chiếc mới trên xa lộ Hà Nội kể từ ngày 1/6/2013.'
        actual = tokenize(text, format="text")
        expected = '( baodautu.vn ) Trao đổi với báo chí chiều 27/5 , ông Lê Quốc Bình , TGĐ Công ty cổ phần đầu tư hạ tầng kỹ thuật TP. HCM ( CII ) chính thức thông tin : UBND TP vừa có quyết định cho phép CII được thu phí hoàn vốn đầu tư dự án xây dựng cầu Rạch Chiếc mới trên xa lộ Hà Nội kể từ ngày 1/6/2013 .'
        self.assertEqual(expected, actual)

    def test_tokenize_tag(self):
        text = '(baodautu.vn) Trao đổi với báo chí chiều 27/5, ông Lê Quốc Bình, TGĐ Công ty cổ phần đầu tư hạ tầng kỹ thuật TP. HCM (CII) chính thức thông tin: UBND TP vừa có quyết định cho phép CII được thu phí hoàn vốn đầu tư dự án xây dựng cầu Rạch Chiếc mới trên xa lộ Hà Nội kể từ ngày 1/6/2013.'
        actual = tokenize(text, tag=True)
        expected = [('(', 'punct'),
                    ('baodautu.vn', 'url'),
                    (')', 'punct'),
                    ('Trao', 'word'),
                    ('đổi', 'word'),
                    ('với', 'word'),
                    ('báo', 'word'),
                    ('chí', 'word'),
                    ('chiều', 'word'),
                    ('27/5', 'datetime'),
                    (',', 'punct'),
                    ('ông', 'word'),
                    ('Lê', 'word'),
                    ('Quốc', 'word'),
                    ('Bình', 'word'),
                    (',', 'punct'),
                    ('TGĐ', 'word'),
                    ('Công', 'word'),
                    ('ty', 'word'),
                    ('cổ', 'word'),
                    ('phần', 'word'),
                    ('đầu', 'word'),
                    ('tư', 'word'),
                    ('hạ', 'word'),
                    ('tầng', 'word'),
                    ('kỹ', 'word'),
                    ('thuật', 'word'),
                    ('TP.', 'abbr'),
                    ('HCM', 'word'),
                    ('(', 'punct'),
                    ('CII', 'word'),
                    (')', 'punct'),
                    ('chính', 'word'),
                    ('thức', 'word'),
                    ('thông', 'word'),
                    ('tin', 'word'),
                    (':', 'sym'),
                    ('UBND', 'word'),
                    ('TP', 'word'),
                    ('vừa', 'word'),
                    ('có', 'word'),
                    ('quyết', 'word'),
                    ('định', 'word'),
                    ('cho', 'word'),
                    ('phép', 'word'),
                    ('CII', 'word'),
                    ('được', 'word'),
                    ('thu', 'word'),
                    ('phí', 'word'),
                    ('hoàn', 'word'),
                    ('vốn', 'word'),
                    ('đầu', 'word'),
                    ('tư', 'word'),
                    ('dự', 'word'),
                    ('án', 'word'),
                    ('xây', 'word'),
                    ('dựng', 'word'),
                    ('cầu', 'word'),
                    ('Rạch', 'word'),
                    ('Chiếc', 'word'),
                    ('mới', 'word'),
                    ('trên', 'word'),
                    ('xa', 'word'),
                    ('lộ', 'word'),
                    ('Hà', 'word'),
                    ('Nội', 'word'),
                    ('kể', 'word'),
                    ('từ', 'word'),
                    ('ngày', 'word'),
                    ('1/6/2013', 'datetime'),
                    ('.', 'punct')]
        self.assertEqual(expected, actual)
