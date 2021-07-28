class BertBinaryClassifier(nn.Module):
    def __init__(self, dropout=0.1):
        super(BertBinaryClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.linear = nn.Linear(768, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, tokens):
        _, pooled_output = self.bert(tokens, utput_all=False)
        linear_output = self.linear(dropout_output)
        proba = self.sigmoid(linear_output)
        return proba
