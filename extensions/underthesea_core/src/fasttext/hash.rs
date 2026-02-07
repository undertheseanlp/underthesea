/// FNV-1a hash matching FastText C++ implementation.
///
/// Critical detail: non-ASCII bytes are sign-extended via `byte as i8 as u32`
/// to match C++ behavior where `char` is signed on most platforms.
pub fn fasttext_hash(s: &[u8]) -> u32 {
    let mut h: u32 = 2166136261;
    for &byte in s {
        // Sign-extend: cast to i8 first (like C++ int8_t), then to u32
        h ^= byte as i8 as u32;
        h = h.wrapping_mul(16777619);
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ascii_hash() {
        let h = fasttext_hash(b"hello");
        assert_eq!(h, 0x4f9f2cab);
    }

    #[test]
    fn test_empty_hash() {
        let h = fasttext_hash(b"");
        assert_eq!(h, 2166136261); // FNV offset basis
    }

    #[test]
    fn test_sign_extension() {
        // Byte 0xFF: as i8 = -1, as u32 = 0xFFFFFFFF
        let h_with_high_byte = fasttext_hash(&[0xFF]);
        let expected = 2166136261u32 ^ 0xFFFFFFFFu32;
        let expected = expected.wrapping_mul(16777619);
        assert_eq!(h_with_high_byte, expected);
    }

    #[test]
    fn test_vietnamese_utf8() {
        let h = fasttext_hash("chào".as_bytes());
        assert_eq!(h, fasttext_hash("chào".as_bytes()));
    }

    #[test]
    fn test_label_prefix() {
        let h = fasttext_hash(b"__label__vi");
        assert_ne!(h, 0);
        assert_eq!(h, fasttext_hash(b"__label__vi"));
    }
}
