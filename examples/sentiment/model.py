import torch
from torch import nn
from transformers import BertForSequenceClassification, BertPreTrainedModel


class BertForSequenceClassifier(BertPreTrainedModel):
    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.bert = BertForSequenceClassification.from_pretrained('bert-base-uncased', config=config)
        self.num_labels = config.num_labels

    # def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
    #             start_positions=None, end_positions=None):
    #     outputs = self.bert(input_ids, attention_mask=attention_mask)
    # #     cls_output = torch.cat((outputs[2][-1][:, 0, ...], outputs[2][-2][:, 0, ...], outputs[2][-3][:, 0, ...],
    # #                             outputs[2][-4][:, 0, ...]), -1)
    # #     logits = self.qa_outputs(cls_output)
    # #     return logits
    #     return outputs
