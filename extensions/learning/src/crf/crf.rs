pub struct Token {
    pub id: String,
    pub token: String,
    pub pos_tag: String
}

pub struct Sentence {
    pub text: String,
    pub tokens: Vec<Token>
}