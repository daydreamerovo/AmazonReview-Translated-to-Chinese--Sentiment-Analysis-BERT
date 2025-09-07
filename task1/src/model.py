import torch
from torch import nn
from transformers import AutoModel
from src import config

class SentimentClassifier(nn.Module):
    def __init__(self, n_classes):
        super(SentimentClassifier, self).__init__()

        self.bert = AutoModel.from_pretrained(config.PRE_TRAINED_MODEL_NAME)
        self.dropout = nn.Dropout(p=0.4)
        self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(
            input_ids = input_ids,
            attention_mask = attention_mask
        )

        pooler_output = bert_output.pooler_output # 输入序列第一个[CLS] token对应的最终输出向量
        # apply regularization
        output = self.dropout(pooler_output)
        # get the classification
        output = self.classifier(output)

        return output