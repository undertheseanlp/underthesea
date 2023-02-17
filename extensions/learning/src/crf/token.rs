pub struct Token {
    pub id: String,
    pub token: String
}

impl Token {
    pub fn new(id: &str, token: &str) -> Token {
        Token {
            id: id.to_string(),
            token: token.to_string(),
        }
    }
}