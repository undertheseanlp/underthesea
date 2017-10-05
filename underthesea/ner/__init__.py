from underthesea import chunk
from underthesea.ner.ner_crf import NERCRFModel


def ner(text, format=None):
    """
    location and classify named entities in text

    :param text: raw text
    :param format:
    :return: list
    """
    text = chunk(text)
    model = NERCRFModel.Instance()
    result = model.predict(text, format)
    return result

if __name__ == '__main__':
    result = ner("Tổng thống Donald Trump và phu nhân Melania Trump ngày 4/10 đã tới Las Vegas thăm hỏi các nạn nhân vụ xả súng đẫm máu vừa xảy ra tại thành phố này và mời họ tới Nhà Trắng.")
    result = ner("Sau “lạm phát”, Sở Nội vụ Hà Nội còn 4 Phó Giám đốc")
    print(0)
