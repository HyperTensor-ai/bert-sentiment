import transformers
import torch.nn as nn
from config import BERT_PATH


class BERTBaseUncased(nn.Module):
    def __init__(self):
        super(BERTBaseUncased, self).__init__()
        self.bert = transformers.BertModel.from_pretrained(BERT_PATH)
        self.bert_drop = nn.Dropout(0.3)
        self.out = nn.Linear(768, 1) # Bert outputs 768 features

    def forward(self, ids, mask, token_type_ids):
        # _ -> Sequence of hidden states for each and every token (512,768) 768 sized vector for each word
        # o2 -> Vector of 768 for each sample in the batch
        _, o2 = self.bert(
            ids, 
            attention_mask=mask, 
            token_type_ids=token_type_ids,
            return_dict=False,
        )

        bo = self.bert_drop(o2)
        output = self.out(bo)
        return output

