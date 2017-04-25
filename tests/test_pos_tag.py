# -*- coding: utf-8 -*-
from unittest import TestCase
from underthesea.pos_tag.predict import predict


class TestPredict(TestCase):
    def test_make_output1(self):
        sentence = u"Chủ nhiệm UB Đối ngoại của Quốc hội Nguyễn Văn Giàu phân tích, ở nước ngoài, ông từng chứng kiến việc doanh nghiệp tặng xe, phương tiện cho tổ chức, nhưng ở Việt Nam, việc này dường như có nhiều… lắt léo."
        actual = predict(sentence, True)
        expected = [(1, u'Chủ nhiệm', 'Nc'), (2, u'UB', 'Np'), (3, u'Đối ngoại', 'Np'),
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
        self.assertEqual(actual, expected)

    def test_make_output_2(self):
        sentence = u'Chiều 20/4, Chủ tịch TP Hà Nội Nguyễn Đức Chung đã về huyện Mỹ Đức giải quyết vụ việc đang diễn ra tại xã Đồng Tâm, sẵn sàng lắng nghe tâm tư bà con'
        actual = predict(sentence, True)
        expected = [(1, u'Chiều', 'N'), (2, u'20', 'M'), (3, u'/', 'CH'), (4, u'4', 'M'), (5, u',', 'CH'),
                    (6, u'Chủ tịch', 'Np'), (7, u'TP', 'Ny'), (8, u'Hà Nội', 'Np'), (9, u'Nguyễn Đức Chung', 'Np'),
                    (10, u'đã', 'R'), (11, u'về', 'V'), (12, u'huyện', 'N'), (13, u'Mỹ Đức', 'Np'),
                    (14, u'giải quyết', 'V'), (15, u'vụ việc', 'N'), (16, u'đang', 'R'), (17, u'diễn', 'V'),
                    (18, u'ra', 'V'), (19, u'tại', 'E'), (20, u'xã', 'N'), (21, u'Đồng Tâm', 'Np'), (22, u',', 'CH'),
                    (23, u'sẵn sàng', 'N'), (24, u'lắng nghe', 'N'), (25, u'tâm tư', 'V'), (26, u'bà', 'N'),
                    (27, u'con', 'N')]
        self.assertEqual(actual, expected)

    def test_make_output_3(self):
        sentence = u'Real Madrid và Atletico Madrid đã hai lần góp mặt ở trận chung kết Champions League'
        actual = predict(sentence, True)
        expected = [(1, u'Real Madrid', 'N'), (2, u'và', 'Cc'), (3, u'Atletico Madrid', 'Np'), (4, u'đã', 'R'),
                    (5, u'hai', 'M'), (6, u'lần', 'N'), (7, u'góp mặt', 'V'), (8, u'ở', 'E'), (9, u'trận', 'N'),
                    (10, u'chung kết', 'N'), (11, u'Champions League', 'Np')]
        self.assertEqual(actual, expected)

    def test_make_output_4(self):
        sentence = u"Chủ nhiệm UB Đối ngoại của Quốc hội Nguyễn Văn Giàu phân tích, ở nước ngoài, ông từng chứng kiến việc doanh nghiệp tặng xe, phương tiện cho tổ chức, nhưng ở Việt Nam, việc này dường như có nhiều… lắt léo."
        actual = predict(sentence, False)
        expected = u'Chủ nhiệm/Nc UB/Np Đối ngoại/Np của/E Quốc hội/Np Nguyễn Văn Giàu/Np phân tích/V ,/CH ở/E nước ngoài/N ,/CH ông/N từng/R chứng kiến/V việc/N doanh nghiệp/V tặng/V xe/N ,/CH phương tiện/V cho/E tổ chức/N ,/CH nhưng/C ở/E Việt Nam/Np ,/CH việc/N này/P dường như/N có/V nhiều/A …/CH lắt léo/V ./CH'
        self.assertEqual(actual, expected)

    def test_make_output_5(self):
        sentence = u"Sang mùa này, hai CLB thành Madrid vẫn thi đấu vô cùng ấn tượng. Họ tiếp tục song hành lọt vào vòng bán kết Champions League. Trong khi Real Madrid xuất sắc vượt qua Bayern Munich với tổng tỷ số 6-3 sau hai lượt đấu thì Atletico Madrid đã loại “hiện tượng” Leicester City với tổng tỷ số 2-1"
        actual = predict(sentence, False)
        expected = u'Sang/V mùa/N này/P ,/CH hai/M CLB/Ny thành/V Madrid/Np vẫn/R thi đấu/V vô cùng/V ấn tượng/N ./CH Họ/Np tiếp tục/V song hành/N lọt/V vào/E vòng/N bán kết/N Champions League/Np ./CH Trong/E khi/N Real Madrid/Np xuất sắc/V vượt/V qua/V Bayern Munich/Np với/E tổng/N tỷ số/N 6/M -/CH 3/M sau/E hai/M lượt/N đấu/V thì/C Atletico Madrid/Np đã/R loại/Nc “/CH hiện tượng/N ”/CH Leicester City/Np với/E tổng/N tỷ số/N 2/M -/CH 1/M'
        self.assertEqual(actual, expected)

    def test_make_output_6(self):
        sentence = u'Theo dự kiến, lễ bốc thăm vòng bán kết Champions League sẽ diễn ra vào lúc 18h00 ngày thứ Sáu (21/4) tại Nyon, Thụy Sĩ. Cũng như vòng tứ kết, vòng đấu này không có đội hạt giống, các CLB cùng quốc gia phải đụng độ với nhau.'
        actual = predict(sentence, False)
        expected = u'Theo/E dự kiến/N ,/CH lễ/N bốc/V thăm/V vòng/N bán kết/N Champions League/Np sẽ/R diễn/V ra/R vào/V lúc/N 18h00/V ngày/N thứ Sáu/N (/CH 21/M //CH 4/M )/CH tại/E Nyon/Np ,/CH Thụy Sĩ/Np ./CH Cũng/Np như/C vòng/N tứ kết/N ,/CH vòng/N đấu/V này/P không/R có/V đội/N hạt giống/N ,/CH các/L CLB/Ny cùng/A quốc gia/N phải/V đụng độ/V với/E nhau/N ./CH'
        self.assertEqual(actual, expected)
