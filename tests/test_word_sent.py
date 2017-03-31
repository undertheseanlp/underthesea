# -*- coding: utf-8 -*-
from unittest import TestCase
from underthesea import word_sent


class TestWord_sent(TestCase):
    def test_word_sent_1(self):
        sentence = u"cộng hòa xã hội chủ nghĩa"
        predict_sentence = word_sent(sentence)
        actual_sentence = u'cộng hòa xã_hội_chủ_nghĩa'
        self.assertEqual(actual_sentence, predict_sentence)

    def test_word_sent_2(self):
        sentence = u"nhật tinh anh đang làm gì đấy ?"
        predict_sentence = word_sent(sentence)
        actual_sentence = u'nhật tinh_anh đang làm gì đấy ?'
        self.assertEqual(actual_sentence, predict_sentence)

    def test_word_sent_3(self):
        sentence = u"hươu rất sợ tiếng động lạ ? ?"
        predict_sentence = word_sent(sentence)
        actual_sentence = u'hươu rất sợ tiếng_động lạ ? ?'
        self.assertEqual(actual_sentence, predict_sentence)

    def test_word_sent_4(self):
        sentence = u"Chúng ta thường nói đến Rau sạch , Rau an toàn để phân biệt với các rau bình thường bán ngoài chợ ."
        predict_sentence = word_sent(sentence)
        actual_sentence = u'Chúng_ta thường nói đến Rau_sạch , Rau an_toàn để phân_biệt với các rau bình_thường bán ngoài chợ .'
        self.assertEqual(actual_sentence, predict_sentence)

    def test_word_sent_5(self):
        sentence = u"Theo thông báo kết luận thanh tra của UBND tỉnh Thanh Hoá sáng nay 30/3, giai đoạn 2010-2015 Sở Xây dựng Thanh Hoá đã bổ nhiệm một số trưởng phòng, phó phòng chưa có trình độ Trung cấp lý luận chính trị, chưa qua lớp bồi dưỡng nghiệp vụ quản lý nhà nước, không đúng quy định của UBND tỉnh Thanh Hoá."
        predict_sentence = word_sent(sentence)
        actual_sentence = u'Theo thông_báo kết_luận thanh_tra của UBND tỉnh Thanh_Hoá sáng nay 30/3, giai_đoạn 2010-2015 Sở_Xây_dựng Thanh_Hoá đã bổ_nhiệm một_số trưởng phòng, phó_phòng chưa có trình_độ Trung_cấp lý_luận chính trị, chưa qua lớp bồi_dưỡng nghiệp_vụ quản_lý nhà_nước, không đúng quy_định của UBND tỉnh Thanh_Hoá.'
        self.assertEqual(actual_sentence, predict_sentence)

    def test_word_sent_6(self):
        sentence = u"Tập thể lãnh đạo Sở Xây dựng không thực hiện nghiêm túc việc đánh giá toàn diện cán bộ trước khi đưa vào quy hoạch, tạo dư luận không tốt. Việc chưa báo cáo về Sở Nội vụ và không công khai việc bà Trần Vũ Quỳnh Anh thôi việc ngày 23/9/2016 thuộc trách nhiệm của Giám đốc Sở Xây dựng tỉnh Thanh Hoá."
        predict_sentence = word_sent(sentence)
        actual_sentence = u'Tập_thể lãnh_đạo Sở_Xây_dựng không thực_hiện nghiêm_túc việc đánh_giá toàn_diện cán_bộ trước khi đưa vào quy hoạch, tạo dư_luận không tốt. Việc chưa báo_cáo về Sở_Nội_vụ và không công_khai việc bà Trần Vũ_Quỳnh Anh thôi_việc ngày 23/9/2016 thuộc trách_nhiệm của Giám_đốc Sở_Xây_dựng tỉnh Thanh_Hoá.'
        self.assertEqual(actual_sentence, predict_sentence)

    def test_word_sent_7(self):
        sentence = u"Có lẽ không ở đâu trên khắp thế giới bóng đá có giải vô địch quốc gia chịu chơi, chịu chi như giải nhà nghề Trung Quốc."
        predict_sentence = word_sent(sentence)
        actual_sentence = u'Có_lẽ không ở đâu trên khắp thế_giới bóng_đá có giải vô_địch_quốc_gia chịu chơi, chịu chi như giải nhà_nghề Trung_Quốc.'
        self.assertEqual(actual_sentence, predict_sentence)

    def test_word_sent_8(self):
        sentence = u"Số những ngôi sao về chiều gia nhập giải nhà nghề Trung Quốc có Carlos Tevez (Argentina) , Graziano Pelle (Italia) , Obafemi Martins (Nigeria) ."
        predict_sentence = word_sent(sentence)
        actual_sentence = u'Số những ngôi_sao về chiều gia_nhập giải nhà_nghề Trung_Quốc có Carlos_Tevez_(Argentina) , Graziano_Pelle_(Italia) , Obafemi_Martins_(Nigeria) .'
        self.assertEqual(actual_sentence, predict_sentence)

    def test_word_sent_9(self):
        sentence = u"Liên quan đến sự việc trên, sáng 30/3, trao đổi với phóng viên Dân trí, Thiếu tá Nguyễn Anh Dũng - Phó trưởng Công an TP Vĩnh Yên (Vĩnh Phúc) - cho biết, ngay khi nhận được thông tin, đơn vị này đã vào cuộc điều tra và xác định người đánh đập cháu H. là ông Hoàng Văn L. (bố đẻ của cháu H.)."
        predict_sentence = word_sent(sentence)
        actual_sentence = u'Liên_quan đến sự_việc trên, sáng 30/3, trao_đổi với phóng_viên Dân_trí, Thiếu_tá Nguyễn_Anh_Dũng - Phó_trưởng Công_an TP Vĩnh_Yên_(Vĩnh_Phúc) - cho biết, ngay khi nhận được thông tin, đơn_vị này đã vào cuộc điều_tra và xác_định người đánh_đập cháu H. là ông Hoàng_Văn_L. (bố đẻ của cháu H.).'
        self.assertEqual(actual_sentence, predict_sentence)

    def test_word_sent_10(self):
        sentence = u" Phát biểu trước báo giới, cựu tiền vệ MU, Paul Scholes đánh giá MU đứng trước cơ hội lớn vô địch Europa League"
        predict_sentence = word_sent(sentence)
        actual_sentence = u'Phát_biểu trước báo giới, cựu tiền_vệ MU, Paul_Scholes đánh_giá MU đứng trước cơ_hội lớn vô_địch Europa_League'
        self.assertEqual(actual_sentence, predict_sentence)

    def test_word_sent_11(self):
        sentence = u"Hơn 2 tỷ USD vốn đầu tư này sẽ được huy động từ hai nguồn chính: Cụ thể, vốn góp của nhà đầu tư là Công ty CP Mặt Trời Vân Đồn (thuộc Sun Group) khoảng 7.125 tỷ đồng, tương đương 312,5 triệu USD, chiếm 15% tổng vốn đầu tư. 85% tổng vốn đầu tư còn lại (khoảng 40.356,9 tỷ đồng) sẽ vay từ Ngân hàng TMCP Công Thương Việt Nam (VietinBank) – chi nhánh Hà Nội theo tiến độ dự kiến từ quý II/2017 đến hết quý IV/2021."
        predict_sentence = word_sent(sentence)
        actual_sentence = u'Hơn 2 tỷ USD vốn đầu_tư này sẽ được huy_động từ hai nguồn chính: Cụ thể, vốn góp của nhà_đầu_tư là Công_ty CP Mặt_Trời Vân_Đồn (thuộc Sun_Group) khoảng 7.125 tỷ đồng, tương_đương 312,5 triệu USD, chiếm 15% tổng vốn đầu tư. 85% tổng vốn đầu_tư còn lại (khoảng 40.356,9 tỷ đồng) sẽ vay từ Ngân_hàng TMCP_Công_Thương_Việt_Nam (VietinBank) – chi_nhánh Hà_Nội theo tiến_độ dự_kiến từ quý II/2017 đến hết quý IV/2021.'
        self.assertEqual(actual_sentence, predict_sentence)

    def test_word_sent_12(self):
        sentence = u"Tích phân là một khái niệm toán học,và cùng với nghịch đảo của nó vi phân (differentiation) đóng vai trò là 2 phép tính cơ bản và chủ chốt trong lĩnh vực giải tích (calculus)."
        predict_sentence = word_sent(sentence)
        actual_sentence = u'Tích_phân là một khái_niệm toán học,và cùng_với nghịch_đảo của nó vi_phân (differentiation) đóng vai_trò là 2 phép_tính cơ_bản và chủ_chốt trong lĩnh_vực giải_tích (calculus).'
        self.assertEqual(actual_sentence, predict_sentence)

    def test_word_sent_13(self):
        sentence = u"Ý tưởng chủ đạo là tích phân và vi phân là hai phép tính nghịch đảo của nhau . "
        predict_sentence = word_sent(sentence)
        actual_sentence = u'Ý_tưởng chủ_đạo là tích_phân và vi_phân là hai phép_tính nghịch_đảo của nhau .'
        self.assertEqual(actual_sentence, predict_sentence)

    def test_word_sent_14(self):
        sentence = u"Vì ngày lang thang vẫn còn dài nên ta không kéo ga thật sâu"
        predict_sentence = word_sent(sentence)
        actual_sentence = u'Vì ngày lang_thang vẫn còn dài nên ta không kéo ga thật sâu'
        self.assertEqual(actual_sentence, predict_sentence)

    def test_word_sent_15(self):
        sentence = u"Vì vậy, để hiểu thêm về ngôn ngữ của người dân xứ Quảng qua đó hiểu về con người vùng đất này hơn và cũng để bổ sung thêm nguồn tư liệu dạy và học Ngữ văn địa phương, bài viết xin giới thiệu một số từ địa phương so với từ phổ thông (toàn dân) và một số âm địa phương so với âm phổ thông (chuẩn - toàn dân) của người Quảng Nam để các bạn đồng nghiệp và các em học sinh tham khảo"
        predict_sentence = word_sent(sentence)
        actual_sentence = u'Vì vậy, để hiểu thêm về ngôn_ngữ của người dân xứ Quảng qua đó hiểu về con_người vùng_đất này hơn và cũng để bổ_sung thêm nguồn tư_liệu dạy và học Ngữ_văn_địa phương, bài viết xin giới_thiệu một_số từ địa_phương so với từ_phổ_thông (toàn dân) và một_số âm địa_phương so với âm_phổ_thông (chuẩn - toàn dân) của người Quảng_Nam để các bạn đồng_nghiệp và các em học_sinh tham_khảo'
        self.assertEqual(actual_sentence, predict_sentence)
