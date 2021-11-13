
fn generate_feature(_a: String) -> String {
    let result = String::from("Chào bạn");
    return result;
}

pub fn lower(_a: String) -> String {
    return _a.to_lowercase();
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        let b = super::generate_feature(String::from("abc"));
        let expected = String::from("Chào bạn");
        assert_eq!(b, expected);
    }
}

