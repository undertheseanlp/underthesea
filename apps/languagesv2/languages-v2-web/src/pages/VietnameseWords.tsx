export interface VietnameseWord {
    word: string;
    partOfSpeech: string;
    frequency: number;
  }
  
  const VietnameseWords: VietnameseWord[] = [
    { word: "Ăn gì", partOfSpeech: "phrase", frequency: 5 },
    { word: "Âm nhạc", partOfSpeech: "noun", frequency: 6 },
    { word: "Anh trai", partOfSpeech: "noun", frequency: 7 },
    { word: "Ánh sáng", partOfSpeech: "noun", frequency: 5 },
    { word: "An toàn", partOfSpeech: "adjective", frequency: 6 },
    { word: "Áo dài", partOfSpeech: "noun", frequency: 8 },
    { word: "Ân cần", partOfSpeech: "adjective", frequency: 4 },
    { word: "Alo", partOfSpeech: "interjection", frequency: 7 },
    { word: "Ăn uống", partOfSpeech: "verb", frequency: 6 },
    { word: "Âu yếm", partOfSpeech: "adjective", frequency: 5 },
    { word: "Âm thanh", partOfSpeech: "noun", frequency: 6 },
    { word: "Bệnh viện", partOfSpeech: "noun", frequency: 6 },
    { word: "Cảm ơn", partOfSpeech: "verb", frequency: 9 },
    { word: "Chúc mừng", partOfSpeech: "verb", frequency: 5 },
    { word: "Điện thoại", partOfSpeech: "noun", frequency: 8 },
    { word: "Được", partOfSpeech: "verb", frequency: 7 },
    { word: "Giúp tôi", partOfSpeech: "verb", frequency: 6 },
    { word: "Học tập", partOfSpeech: "verb", frequency: 6 },
    { word: "Khách sạn", partOfSpeech: "noun", frequency: 4 },
    { word: "Không", partOfSpeech: "adverb", frequency: 10 },
    { word: "Làm ơn", partOfSpeech: "phrase", frequency: 8 },
    { word: "Máy bay", partOfSpeech: "noun", frequency: 4 },
    { word: "Mua sắm", partOfSpeech: "verb", frequency: 7 },
    { word: "Nghỉ ngơi", partOfSpeech: "verb", frequency: 7 },
    { word: "Nhà hàng", partOfSpeech: "noun", frequency: 5 },
    { word: "Siêu thị", partOfSpeech: "noun", frequency: 6 },
    { word: "Sức khỏe", partOfSpeech: "noun", frequency: 7 },
    { word: "Tạm biệt", partOfSpeech: "interjection", frequency: 7 },
    { word: "Trường học", partOfSpeech: "noun", frequency: 7 },
    { word: "Tốt lắm", partOfSpeech: "adjective", frequency: 4 },
    { word: "Uống nước", partOfSpeech: "verb", frequency: 6 },
    { word: "Vâng", partOfSpeech: "interjection", frequency: 6 },
    { word: "Xe máy", partOfSpeech: "noun", frequency: 6 },
    { word: "Xin chào", partOfSpeech: "interjection", frequency: 8 },
    { word: "Xin lỗi", partOfSpeech: "verb", frequency: 7 },
    { word: "Ô tô", partOfSpeech: "noun", frequency: 5 }
  ];
  
  export default VietnameseWords;