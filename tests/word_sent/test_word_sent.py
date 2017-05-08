# -*- coding: utf-8 -*-
from unittest import TestCase
from underthesea import word_sent


class TestWord_sent(TestCase):
    def test_word_sent_1(self):
        sentence = u"cộng hòa xã hội chủ nghĩa"
        actual = word_sent(sentence)
        expected = [u"cộng", u"hòa", u"xã hội chủ nghĩa"]
        self.assertEqual(actual, expected)

    def test_word_sent_2(self):
        sentence = u"hươu rất sợ tiếng động lạ ? ?"
        actual = word_sent(sentence)
        expected = [u"hươu", u"rất", u"sợ", u"tiếng động", u"lạ", u"?", u"?"]
        self.assertEqual(actual, expected)

    def test_word_sent_3(self):
        sentence = u"Chúng ta thường nói đến Rau sạch, Rau an toàn để phân biệt với các rau bình thường bán ngoài chợ ."
        actual = word_sent(sentence)
        expected = [u"Chúng ta", u"thường", u"nói", u"đến", u"Rau sạch", u",", u"Rau", u"an toàn", u"để", u"phân biệt",
                    u"với", u"các", u"rau", u"bình thường", u"bán", u"ngoài", u"chợ", u"."]
        self.assertEqual(actual, expected)

    def test_word_sent_3_1(self):
        sentence = u"Chúng ta thường nói đến Rau sạch, Rau an toàn để phân biệt với các rau bình thường bán ngoài chợ ."
        actual = word_sent(sentence, format="text")
        expected = u'Chúng_ta thường nói đến Rau_sạch , Rau an_toàn để phân_biệt với các rau bình_thường bán ngoài chợ .'
        self.assertEqual(actual, expected)

    def test_word_sent_4(self):
        sentence = u"Theo thông báo kết luận thanh tra của UBND tỉnh Thanh Hoá sáng nay 30/3, giai đoạn 2010-2015 Sở Xây dựng Thanh Hoá đã bổ nhiệm một số trưởng phòng, phó phòng chưa có trình độ Trung cấp lý luận chính trị, chưa qua lớp bồi dưỡng nghiệp vụ quản lý nhà nước, không đúng quy định của UBND tỉnh Thanh Hoá."
        actual = word_sent(sentence)
        expected = [u"Theo", u"thông báo", u"kết luận", u"thanh tra", u"của", u"UBND", u"tỉnh", u"Thanh Hoá", u"sáng",
                    u"nay", u"30", u"/", u"3", u",", u"giai đoạn", u"2010", u"-", u"2015", u"Sở Xây dựng", u"Thanh Hoá",
                    u"đã", u"bổ nhiệm", u"một số", u"trưởng phòng", u",", u"phó phòng", u"chưa", u"có", u"trình độ",
                    u"Trung cấp", u"lý luận", u"chính trị", u",", u"chưa", u"qua", u"lớp", u"bồi dưỡng", u"nghiệp vụ",
                    u"quản lý", u"nhà nước", u",", u"không", u"đúng", u"quy định", u"của", u"UBND", u"tỉnh",
                    u"Thanh Hoá", u"."
                    ]
        self.assertEqual(actual, expected)

    def test_word_sent_5(self):
        sentence = u"Tập thể lãnh đạo Sở Xây dựng không thực hiện nghiêm túc việc đánh giá toàn diện cán bộ trước khi đưa vào quy hoạch, tạo dư luận không tốt. Việc chưa báo cáo về Sở Nội vụ và không công khai việc bà Trần Vũ Quỳnh Anh thôi việc ngày 23/9/2016 thuộc trách nhiệm của Giám đốc Sở Xây dựng tỉnh Thanh Hoá."
        actual = word_sent(sentence)
        expected = [u"Tập thể", u"lãnh đạo", u"Sở Xây dựng", u"không", u"thực hiện", u"nghiêm túc", u"việc",
                    u"đánh giá", u"toàn diện", u"cán bộ", u"trước", u"khi", u"đưa", u"vào", u"quy hoạch", u",", u"tạo",
                    u"dư luận", u"không", u"tốt", u".", u"Việc", u"chưa", u"báo cáo", u"về", u"Sở Nội vụ", u"và",
                    u"không", u"công khai", u"việc", u"bà", u"Trần", u"Vũ Quỳnh", u"Anh", u"thôi việc", u"ngày", u"23",
                    u"/", u"9", u"/", u"2016", u"thuộc", u"trách nhiệm", u"của", u"Giám đốc", u"Sở Xây dựng", u"tỉnh",
                    u"Thanh Hoá", u"."
                    ]
        self.assertEqual(actual, expected)

    def test_word_sent_6(self):
        sentence = u"Có lẽ không ở đâu trên khắp thế giới bóng đá có giải vô địch quốc gia chịu chơi, chịu chi như giải nhà nghề Trung Quốc."
        actual = word_sent(sentence)
        expected = [u"Có lẽ", u"không", u"ở", u"đâu", u"trên", u"khắp", u"thế giới", u"bóng đá", u"có", u"giải",
                    u"vô địch quốc gia", u"chịu chơi", u",", u"chịu", u"chi", u"như", u"giải", u"nhà nghề",
                    u"Trung Quốc", u"."]
        self.assertEqual(actual, expected)

    def test_word_sent_7(self):
        sentence = u"Số những ngôi sao về chiều gia nhập giải nhà nghề Trung Quốc có Carlos Tevez (Argentina), Graziano Pelle (Italia), Obafemi Martins (Nigeria) ."
        actual = word_sent(sentence)
        expected = [u"Số", u"những", u"ngôi sao", u"về", u"chiều", u"gia nhập", u"giải", u"nhà nghề", u"Trung Quốc",
                    u"có", u"Carlos Tevez", u"(", u"Argentina", u")", u",", u"Graziano Pelle", u"(", u"Italia", u")",
                    u",", u"Obafemi Martins", u"(", u"Nigeria", u")", u"."]
        self.assertEqual(actual, expected)

    def test_word_sent_8(self):
        sentence = u"Liên quan đến sự việc trên, sáng 30/3, trao đổi với phóng viên Dân trí, Thiếu tá Nguyễn Anh Dũng - Phó trưởng Công an TP Vĩnh Yên (Vĩnh Phúc) - cho biết, ngay khi nhận được thông tin, đơn vị này đã vào cuộc điều tra và xác định người đánh đập cháu H. là ông Hoàng Văn L. (bố đẻ của cháu H.)."
        actual = word_sent(sentence)
        expected = [u"Liên quan", u"đến", u"sự việc", u"trên", u",", u"sáng", u"30", u"/", u"3", u",", u"trao đổi",
                    u"với", u"phóng viên", u"Dân trí", u",", u"Thiếu tá", u"Nguyễn Anh Dũng", u"-", u"Phó trưởng",
                    u"Công an", u"TP", u"Vĩnh Yên", u"(", u"Vĩnh Phúc", u")", u"-", u"cho biết", u",", u"ngay", u"khi",
                    u"nhận", u"được", u"thông tin", u",", u"đơn vị", u"này", u"đã", u"vào", u"cuộc", u"điều tra", u"và",
                    u"xác định", u"người", u"đánh đập", u"cháu", u"H", u".", u"là", u"ông", u"Hoàng Văn L", u".", u"(",
                    u"bố đẻ", u"của", u"cháu", u"H", u".", u")", u"."
                    ]
        self.assertEqual(actual, expected)

    def test_word_sent_9(self):
        sentence = u"Phát biểu trước báo giới, cựu tiền vệ MU, Paul Scholes đánh giá MU đứng trước cơ hội lớn vô địch Europa League"
        actual = word_sent(sentence)
        expected = [u"Phát biểu", u"trước", u"báo giới", u",", u"cựu", u"tiền vệ", u"MU", u",", u"Paul Scholes",
                    u"đánh giá", u"MU", u"đứng", u"trước", u"cơ hội", u"lớn", u"vô địch", u"Europa League"]
        self.assertEqual(actual, expected)

    def test_word_sent_10(self):
        sentence = u"Hơn 2 tỷ USD vốn đầu tư này sẽ được huy động từ hai nguồn chính: Cụ thể, vốn góp của nhà đầu tư là Công ty CP Mặt Trời Vân Đồn (thuộc Sun Group) khoảng 7.125 tỷ đồng, tương đương 312,5 triệu USD, chiếm 15% tổng vốn đầu tư. 85% tổng vốn đầu tư còn lại (khoảng 40.356,9 tỷ đồng) sẽ vay từ Ngân hàng TMCP Công Thương Việt Nam (VietinBank) – chi nhánh Hà Nội theo tiến độ dự kiến từ quý II/2017 đến hết quý IV/2021."
        actual = word_sent(sentence)
        expected = [
            u"Hơn", u"2", u"tỷ", u"USD", u"vốn", u"đầu tư", u"này", u"sẽ", u"được", u"huy động", u"từ", u"hai",
            u"nguồn", u"chính", u":", u"Cụ thể", u",", u"vốn", u"góp", u"của", u"nhà đầu tư", u"là", u"Công ty", u"CP",
            u"Mặt Trời Vân Đồn", u"(", u"thuộc", u"Sun Group", u")", u"khoảng", u"7.125", u"tỷ", u"đồng", u",",
            u"tương đương", u"312,5", u"triệu", u"USD", u",", u"chiếm", u"15", u"%", u"tổng", u"vốn", u"đầu tư", u".",
            u"85", u"%", u"tổng", u"vốn", u"đầu tư", u"còn", u"lại", u"(", u"khoảng", u"40.356,9", u"tỷ", u"đồng", u")",
            u"sẽ", u"vay", u"từ", u"Ngân hàng", u"TMCP Công Thương Việt Nam", u"(", u"VietinBank", u")", u"–",
            u"chi nhánh", u"Hà Nội", u"theo", u"tiến độ", u"dự kiến", u"từ", u"quý", u"II", u"/", u"2017", u"đến",
            u"hết", u"quý", u"IV", u"/", u"2021", u"."
        ]
        self.assertEqual(actual, expected)

    def test_word_sent_11(self):
        sentence = u"Tích phân là một khái niệm toán học,và cùng với nghịch đảo của nó vi phân (differentiation) đóng vai trò là 2 phép tính cơ bản và chủ chốt trong lĩnh vực giải tích (calculus)."
        actual = word_sent(sentence)
        expected = [u"Tích phân", u"là", u"một", u"khái niệm", u"toán học", u",", u"và", u"cùng với", u"nghịch đảo",
                    u"của", u"nó", u"vi phân", u"(", u"differentiation", u")", u"đóng", u"vai trò", u"là", u"2",
                    u"phép tính", u"cơ bản", u"và", u"chủ chốt", u"trong", u"lĩnh vực", u"giải tích", u"(", u"calculus",
                    u")", u"."
                    ]
        self.assertEqual(actual, expected)

    def test_word_sent_12(self):
        sentence = u"Ý tưởng chủ đạo là tích phân và vi phân là hai phép tính nghịch đảo của nhau . "
        actual = word_sent(sentence)
        expected = [u"Ý tưởng", u"chủ đạo", u"là", u"tích phân", u"và", u"vi phân", u"là", u"hai", u"phép tính",
                    u"nghịch đảo", u"của", u"nhau", u"."]
        self.assertEqual(actual, expected)

    def test_word_sent_13(self):
        sentence = u"Vì ngày lang thang vẫn còn dài nên ta không kéo ga thật sâu"
        actual = word_sent(sentence)
        expected = [u"Vì", u"ngày", u"lang thang", u"vẫn", u"còn", u"dài", u"nên", u"ta", u"không", u"kéo", u"ga",
                    u"thật", u"sâu"]
        self.assertEqual(actual, expected)

    def test_word_sent_14(self):
        sentence = u"Vì vậy, để hiểu thêm về ngôn ngữ của người dân xứ Quảng qua đó hiểu về con người vùng đất này hơn và cũng để bổ sung thêm nguồn tư liệu dạy và học Ngữ văn địa phương, bài viết xin giới thiệu một số từ địa phương so với từ phổ thông (toàn dân) và một số âm địa phương so với âm phổ thông (chuẩn - toàn dân) của người Quảng Nam để các bạn đồng nghiệp và các em học sinh tham khảo"
        actual = word_sent(sentence)
        expected = [
            u"Vì vậy", u",", u"để", u"hiểu", u"thêm", u"về", u"ngôn ngữ", u"của", u"người", u"dân", u"xứ", u"Quảng",
            u"qua", u"đó", u"hiểu", u"về", u"con người", u"vùng đất", u"này", u"hơn", u"và", u"cũng", u"để", u"bổ sung",
            u"thêm", u"nguồn", u"tư liệu", u"dạy", u"và", u"học", u"Ngữ văn", u"địa phương", u",", u"bài", u"viết",
            u"xin", u"giới thiệu", u"một số", u"từ", u"địa phương", u"so", u"với", u"từ phổ thông", u"(", u"toàn dân",
            u")", u"và", u"một số", u"âm", u"địa phương", u"so", u"với", u"âm phổ thông", u"(", u"chuẩn", u"-",
            u"toàn dân", u")", u"của", u"người", u"Quảng Nam", u"để", u"các", u"bạn", u"đồng nghiệp", u"và", u"các",
            u"em", u"học sinh", u"tham khảo"
        ]
        self.assertEqual(actual, expected)
