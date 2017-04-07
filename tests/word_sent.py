# -*- coding: utf-8 -*-
from unittest import TestCase
from underthesea import word_sent


class TestWord_sent(TestCase):
    def test_tokenize(self):
        text = u"""Tổng thống Nga coi việc Mỹ không kích căn cứ quân sự của Syria là "sự gây hấn nhằm vào một quốc gia có chủ quyền", gây tổn hại đến quan hệ Moscow-Washington."""
        text = word_sent(text)
        expected = u"""Tổng_thống Nga_coi việc Mỹ không_kích căn_cứ quân_sự của Syria là " sự gây hấn nhằm vào một quốc_gia có chủ_quyền " , gây tổn_hại đến quan_hệ Moscow - Washington ."""
        self.assertEqual(text, expected)

    def test_tokenize_2(self):
        text = u"""Vào tháng 10 năm ngoái, nghi phạm này từng bị bắt khi ăn cắp 25 mỹ phẩm trị giá 24.000 yen (hơn 200 USD) tại một cửa hàng giảm giá ở tỉnh Hyogo. Cùng năm, 4 thành viên khác cũng bị bắt vì ăn cắp một số lượng lớn son dưỡng môi và mỹ phẩm tại các cửa hiệu ở Osaka. Sau khi bị truy tố, 3 người trong số này được hưởng án treo."""
        text = word_sent(text)
        expected = u'Vào tháng 10 năm ngoái , nghi phạm này từng bị bắt khi ăn_cắp 25 mỹ phẩm trị_giá 24.000 yen ( hơn 200 USD ) tại một cửa_hàng giảm_giá ở tỉnh Hyogo . Cùng năm , 4 thành_viên khác cũng bị bắt vì ăn_cắp một số_lượng lớn son_dưỡng môi và mỹ phẩm tại các cửa_hiệu ở Osaka . Sau khi bị truy_tố , 3 người trong số này được hưởng án_treo .'
        self.assertEqual(text, expected)

    def test_tokenize_3(self):
        text = u"""Giá khuyến mãi: 140.000đ / kg  ==> giảm được 20%"""
        text = word_sent(text)
        expected = u'Giá khuyến_mãi : 140.000 đ / kg ==> giảm được 20 %'
        self.assertEqual(text, expected)

    def test_tokenize_4(self):
        text = u"""Microsoft hôm nay đã công bố cấu hình chính thức của thế hệ Xbox tiếp theo, tên mã Scorpio, và tin tưởng đây là loại console có sức mạnh xử lý cao nhất hiện giờ, mạnh hơn cả PS4 Pro. Tại buổi demo sản phẩm, Xbox Scorpio có thể chơi game đua xe Forza độ phân giải 4K với tốc độ đạt tới 60 khung hình/giây. Đồng thời Xbox Scorpio cũng sẽ tương thích với các tựa game Xbox cũ và quan trọng hơn nữa đó là các tựa game đó sẽ có đồ họa đẹp hơn, FPS cao hơn, thời gian load nhanh hơn khi chạy trên Xbox Scorpio."""
        text = word_sent(text)
        expected = u'Microsoft hôm_nay đã công_bố cấu_hình chính_thức của thế_hệ Xbox tiếp_theo , tên mã Scorpio , và tin_tưởng đây là loại console có sức_mạnh xử_lý cao nhất hiện_giờ , mạnh hơn cả PS4 Pro . Tại buổi demo sản_phẩm , Xbox_Scorpio có_thể chơi game đua xe Forza_độ phân_giải 4K với tốc_độ đạt tới 60 khung hình / giây . Đồng_thời Xbox_Scorpio cũng sẽ tương_thích với các tựa game Xbox cũ và quan_trọng hơn nữa đó là các tựa game đó sẽ có đồ_họa đẹp hơn , FPS cao hơn , thời_gian load nhanh hơn khi chạy trên Xbox_Scorpio .'
        self.assertEqual(text, expected)

    def test_tokenize_5(self):
        text = u"""Tuyên bố của Điện Kremlin cũng nhấn mạnh, vụ tấn công sẽ “hủy hoại nghiêm trọng quan hệ Nga-Mỹ” và tạo ra “trở ngại cực lớn” cho việc tạo lập một liên minh quốc tế chống Tổ chức Nhà nước Hồi giáo tự xưng (IS)."""
        text = word_sent(text)
        expected = u'Tuyên_bố của Điện_Kremlin cũng nhấn_mạnh , vụ tấn_công sẽ “ hủy_hoại nghiêm_trọng quan_hệ Nga - Mỹ ” và tạo ra “ trở_ngại cực lớn ” cho việc tạo_lập một liên_minh quốc_tế chống Tổ_chức Nhà_nước Hồi_giáo tự_xưng ( IS ) .'
        self.assertEqual(text, expected)

    def test_tokenize_6(self):
        text = u"""Hà nội ngày 07/04/2017. Nghĩ mãi không ra cái test nào đau đầu vãi chưởng."""
        text = word_sent(text)
        expected = u'Hà nội ngày 07 / 04 / 2017 . Nghĩ mãi không ra cái test nào đau_đầu vãi chưởng .'
        self.assertEqual(text, expected)
