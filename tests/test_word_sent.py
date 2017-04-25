# -*- coding: utf-8 -*-
from unittest import TestCase
from underthesea import word_sent


class TestWord_sent(TestCase):
    def test_word_sent_1(self):
        sentence = u"cộng hòa xã hội chủ nghĩa"
        actual = word_sent(sentence, True)
        expected = u'cộng hòa xã_hội_chủ_nghĩa'
        self.assertEqual(actual, expected)

    def test_word_sent_2(self):
        sentence = u"hươu rất sợ tiếng động lạ ? ?"
        actual = word_sent(sentence, True)
        expected = u'hươu rất sợ tiếng_động lạ ? ?'
        self.assertEqual(actual, expected)

    def test_word_sent_3(self):
        sentence = u"Chúng ta thường nói đến Rau sạch, Rau an toàn để phân biệt với các rau bình thường bán ngoài chợ ."
        actual = word_sent(sentence, True)
        expected = u'Chúng_ta thường nói đến Rau_sạch , Rau an_toàn để phân_biệt với các rau bình_thường bán ngoài chợ .'
        self.assertEqual(actual, expected)

    def test_word_sent_4(self):
        sentence = u"Theo thông báo kết luận thanh tra của UBND tỉnh Thanh Hoá sáng nay 30/3, giai đoạn 2010-2015 Sở Xây dựng Thanh Hoá đã bổ nhiệm một số trưởng phòng, phó phòng chưa có trình độ Trung cấp lý luận chính trị, chưa qua lớp bồi dưỡng nghiệp vụ quản lý nhà nước, không đúng quy định của UBND tỉnh Thanh Hoá."
        actual = word_sent(sentence, True)
        expected = u'Theo thông_báo kết_luận thanh_tra của UBND tỉnh Thanh_Hoá sáng nay 30 / 3 , giai_đoạn 2010 - 2015 Sở_Xây_dựng Thanh_Hoá đã bổ_nhiệm một_số trưởng_phòng , phó_phòng chưa có trình_độ Trung_cấp lý_luận chính_trị , chưa qua lớp bồi_dưỡng nghiệp_vụ quản_lý nhà_nước , không đúng quy_định của UBND tỉnh Thanh_Hoá .'
        self.assertEqual(actual, expected)

    def test_word_sent_5(self):
        sentence = u"Tập thể lãnh đạo Sở Xây dựng không thực hiện nghiêm túc việc đánh giá toàn diện cán bộ trước khi đưa vào quy hoạch, tạo dư luận không tốt. Việc chưa báo cáo về Sở Nội vụ và không công khai việc bà Trần Vũ Quỳnh Anh thôi việc ngày 23/9/2016 thuộc trách nhiệm của Giám đốc Sở Xây dựng tỉnh Thanh Hoá."
        actual = word_sent(sentence, True)
        expected = u'Tập_thể lãnh_đạo Sở_Xây_dựng không thực_hiện nghiêm_túc việc đánh_giá toàn_diện cán_bộ trước khi đưa vào quy_hoạch , tạo dư_luận không tốt . Việc chưa báo_cáo về Sở_Nội_vụ và không công_khai việc bà Trần Vũ_Quỳnh Anh thôi_việc ngày 23 / 9 / 2016 thuộc trách_nhiệm của Giám_đốc Sở_Xây_dựng tỉnh Thanh_Hoá .'
        self.assertEqual(actual, expected)

    def test_word_sent_6(self):
        sentence = u"Có lẽ không ở đâu trên khắp thế giới bóng đá có giải vô địch quốc gia chịu chơi, chịu chi như giải nhà nghề Trung Quốc."
        actual = word_sent(sentence, True)
        expected = u'Có_lẽ không ở đâu trên khắp thế_giới bóng_đá có giải vô_địch_quốc_gia chịu_chơi , chịu chi như giải nhà_nghề Trung_Quốc .'
        self.assertEqual(actual, expected)

    def test_word_sent_7(self):
        sentence = u"Số những ngôi sao về chiều gia nhập giải nhà nghề Trung Quốc có Carlos Tevez (Argentina), Graziano Pelle (Italia), Obafemi Martins (Nigeria) ."
        actual = word_sent(sentence, True)
        expected = u'Số những ngôi_sao về chiều gia_nhập giải nhà_nghề Trung_Quốc có Carlos_Tevez ( Argentina ) , Graziano_Pelle ( Italia ) , Obafemi_Martins ( Nigeria ) .'
        self.assertEqual(actual, expected)

    def test_word_sent_8(self):
        sentence = u"Liên quan đến sự việc trên, sáng 30/3, trao đổi với phóng viên Dân trí, Thiếu tá Nguyễn Anh Dũng - Phó trưởng Công an TP Vĩnh Yên (Vĩnh Phúc) - cho biết, ngay khi nhận được thông tin, đơn vị này đã vào cuộc điều tra và xác định người đánh đập cháu H. là ông Hoàng Văn L. (bố đẻ của cháu H.)."
        actual = word_sent(sentence, True)
        expected = u'Liên_quan đến sự_việc trên , sáng 30 / 3 , trao_đổi với phóng_viên Dân_trí , Thiếu_tá Nguyễn_Anh_Dũng - Phó_trưởng Công_an TP Vĩnh_Yên ( Vĩnh_Phúc ) - cho_biết , ngay khi nhận được thông_tin , đơn_vị này đã vào cuộc điều_tra và xác_định người đánh_đập cháu H . là ông Hoàng_Văn_L . ( bố_đẻ của cháu H . ) .'
        self.assertEqual(actual, expected)

    def test_word_sent_9(self):
        sentence = u"Phát biểu trước báo giới, cựu tiền vệ MU, Paul Scholes đánh giá MU đứng trước cơ hội lớn vô địch Europa League"
        actual = word_sent(sentence, True)
        expected = u'Phát_biểu trước báo_giới , cựu tiền_vệ MU , Paul_Scholes đánh_giá MU đứng trước cơ_hội lớn vô_địch Europa_League'
        self.assertEqual(actual, expected)

    def test_word_sent_10(self):
        sentence = u"Hơn 2 tỷ USD vốn đầu tư này sẽ được huy động từ hai nguồn chính: Cụ thể, vốn góp của nhà đầu tư là Công ty CP Mặt Trời Vân Đồn (thuộc Sun Group) khoảng 7.125 tỷ đồng, tương đương 312,5 triệu USD, chiếm 15% tổng vốn đầu tư. 85% tổng vốn đầu tư còn lại (khoảng 40.356,9 tỷ đồng) sẽ vay từ Ngân hàng TMCP Công Thương Việt Nam (VietinBank) – chi nhánh Hà Nội theo tiến độ dự kiến từ quý II/2017 đến hết quý IV/2021."
        actual = word_sent(sentence, True)
        expected = u'Hơn 2 tỷ USD vốn đầu_tư này sẽ được huy_động từ hai nguồn chính : Cụ_thể , vốn góp của nhà_đầu_tư là Công_ty CP Mặt_Trời_Vân_Đồn ( thuộc Sun_Group ) khoảng 7.125 tỷ đồng , tương_đương 312,5 triệu USD , chiếm 15 % tổng vốn đầu_tư . 85 % tổng vốn đầu_tư còn lại ( khoảng 40.356,9 tỷ đồng ) sẽ vay từ Ngân_hàng TMCP_Công_Thương_Việt_Nam ( VietinBank ) – chi_nhánh Hà_Nội theo tiến_độ dự_kiến từ quý II / 2017 đến hết quý IV / 2021 .'
        self.assertEqual(actual, expected)

    def test_word_sent_11(self):
        sentence = u"Tích phân là một khái niệm toán học,và cùng với nghịch đảo của nó vi phân (differentiation) đóng vai trò là 2 phép tính cơ bản và chủ chốt trong lĩnh vực giải tích (calculus)."
        actual = word_sent(sentence, True)
        expected = u'Tích_phân là một khái_niệm toán_học , và cùng_với nghịch_đảo của nó vi_phân ( differentiation ) đóng vai_trò là 2 phép_tính cơ_bản và chủ_chốt trong lĩnh_vực giải_tích ( calculus ) .'
        self.assertEqual(actual, expected)

    def test_word_sent_12(self):
        sentence = u"Ý tưởng chủ đạo là tích phân và vi phân là hai phép tính nghịch đảo của nhau . "
        actual = word_sent(sentence, True)
        expected = u'Ý_tưởng chủ_đạo là tích_phân và vi_phân là hai phép_tính nghịch_đảo của nhau .'
        self.assertEqual(actual, expected)

    def test_word_sent_13(self):
        sentence = u"Vì ngày lang thang vẫn còn dài nên ta không kéo ga thật sâu"
        actual = word_sent(sentence, True)
        expected = u'Vì ngày lang_thang vẫn còn dài nên ta không kéo ga thật sâu'
        self.assertEqual(actual, expected)

    def test_word_sent_14(self):
        sentence = u"Vì vậy, để hiểu thêm về ngôn ngữ của người dân xứ Quảng qua đó hiểu về con người vùng đất này hơn và cũng để bổ sung thêm nguồn tư liệu dạy và học Ngữ văn địa phương, bài viết xin giới thiệu một số từ địa phương so với từ phổ thông (toàn dân) và một số âm địa phương so với âm phổ thông (chuẩn - toàn dân) của người Quảng Nam để các bạn đồng nghiệp và các em học sinh tham khảo"
        actual = word_sent(sentence, True)
        expected = u'Vì_vậy , để hiểu thêm về ngôn_ngữ của người dân xứ Quảng qua đó hiểu về con_người vùng_đất này hơn và cũng để bổ_sung thêm nguồn tư_liệu dạy và học Ngữ_văn địa_phương , bài viết xin giới_thiệu một_số từ địa_phương so với từ_phổ_thông ( toàn_dân ) và một_số âm địa_phương so với âm_phổ_thông ( chuẩn - toàn_dân ) của người Quảng_Nam để các bạn đồng_nghiệp và các em học_sinh tham_khảo'
        self.assertEqual(actual, expected)

    def test_list_word_sent_1(self):
        sentence = u"Vì vậy, để hiểu thêm về ngôn ngữ của người dân xứ Quảng qua đó hiểu về con người vùng đất này hơn và cũng để bổ sung thêm nguồn tư liệu dạy và học Ngữ văn địa phương, bài viết xin giới thiệu một số từ địa phương so với từ phổ thông (toàn dân) và một số âm địa phương so với âm phổ thông (chuẩn - toàn dân) của người Quảng Nam để các bạn đồng nghiệp và các em học sinh tham khảo"
        actual = word_sent(sentence, False)
        # expected = [x.encode('utf-8') for x in actual]
        expected = [u'Vì vậy', u',', u'để', u'hiểu', u'thêm', u'về', u'ngôn ngữ',
                    u'của', u'người', u'dân', u'xứ', u'Quảng', u'qua', u'đó',
                    u'hiểu', u'về', u'con người', u'vùng đất', u'này', u'hơn',
                    u'và', u'cũng', u'để', u'bổ sung', u'thêm', u'nguồn',
                    u'tư liệu', u'dạy', u'và', u'học', u'Ngữ văn',
                    u'địa phương', u',', u'bài', u'viết', u'xin', u'giới thiệu',
                    u'một số', u'từ', u'địa phương', u'so', u'với',
                    u'từ phổ thông', u'(', u'toàn dân', u')', u'và', u'một số',
                    u'âm', u'địa phương', u'so', u'với', u'âm phổ thông', u'(',
                    u'chuẩn', u'-', u'toàn dân', u')', u'của', u'người', u'Quảng Nam',
                    u'để', u'các', u'bạn', u'đồng nghiệp', u'và', u'các', u'em',
                    u'học sinh', u'tham khảo']
        self.assertEqual(actual, expected)

    def test_list_word_sent_2(self):
        sentence = u"cộng hòa xã hội chủ nghĩa"
        actual = word_sent(sentence, False)
        expected = [u'cộng', u'hòa', u'xã hội chủ nghĩa']
        self.assertEqual(actual, expected)

    def test_list_word_sent_3(self):
        sentence = u"Theo thông báo kết luận thanh tra của UBND tỉnh Thanh Hoá sáng nay 30/3, giai đoạn 2010-2015 Sở Xây dựng Thanh Hoá đã bổ nhiệm một số trưởng phòng, phó phòng chưa có trình độ Trung cấp lý luận chính trị, chưa qua lớp bồi dưỡng nghiệp vụ quản lý nhà nước, không đúng quy định của UBND tỉnh Thanh Hoá."
        actual = word_sent(sentence, False)
        expected = [u'Theo', u'thông báo', u'kết luận', u'thanh tra', u'của', u'UBND',
                    u'tỉnh', u'Thanh Hoá', u'sáng', u'nay', u'30', u'/', u'3', u',', u'giai đoạn',
                    u'2010', u'-', u'2015', u'Sở Xây dựng', u'Thanh Hoá', u'đã',
                    u'bổ nhiệm', u'một số', u'trưởng phòng', u',', u'phó phòng',
                    u'chưa', u'có', u'trình độ', u'Trung cấp', u'lý luận',
                    u'chính trị', u',', u'chưa', u'qua', u'lớp', u'bồi dưỡng',
                    u'nghiệp vụ', u'quản lý', u'nhà nước', u',', u'không',
                    u'đúng', u'quy định', u'của', u'UBND', u'tỉnh', u'Thanh Hoá', u'.']
        self.assertEqual(actual, expected)

    def test_list_word_sent_4(self):
        sentence = u'Chủ nhiệm UB Đối ngoại của Quốc hội Nguyễn Văn Giàu phân tích, ở nước ngoài, ông từng chứng kiến việc doanh nghiệp tặng xe, phương tiện cho tổ chức, nhưng ở Việt Nam, việc này dường như có nhiều… lắt léo.'
        actual = word_sent(sentence)
        expected = [u'Chủ nhiệm', u'UB', u'Đối ngoại', u'của', u'Quốc hội', u'Nguyễn Văn Giàu', u'phân tích', u',', u'ở', u'nước ngoài', u',', u'ông', u'từng', u'chứng kiến', u'việc', u'doanh nghiệp', u'tặng', u'xe', u',', u'phương tiện', u'cho', u'tổ chức', u',', u'nhưng', u'ở', u'Việt Nam', u',', u'việc', u'này', u'dường như', u'có', u'nhiều', u'…', u'lắt léo', u'.']
        self.assertEquals(actual, expected)
