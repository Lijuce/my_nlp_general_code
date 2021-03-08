# bert模型的基本使用，以RoBerta为例
import torch
from transformers import RobertaModel, RobertaTokenizer

robert_model = RobertaModel.from_pretrained(pretrained_model_name_or_path="roberta-base")
for param in robert_model.parameters():
    param.requires_grad = True

tokenizer_class = RobertaTokenizer
tokenizer = tokenizer_class.from_pretrained(pretrained_model_name_or_path="roberta-base", cache_dir='.')

sentence = "The location that appointed Beatrix of the Netherlands to governmental position uses what type of money"
sentence = "<s> " + sentence + " </s>"
sent_tokenized = tokenizer.tokenize(sentence)
sent_tokenized = torch.tensor(tokenizer.encode(sent_tokenized, add_special_tokens=False))

all_sent = [sent_tokenized, sent_tokenized, sent_tokenized]
all_sent = torch.stack(all_sent, dim=0)

attention_mask = []
for s in sent_tokenized:
    if s == 1:
        attention_mask.append(0)
    else:
        attention_mask.append(1)
attention_mask = torch.tensor(attention_mask, dtype=torch.long)

all_att_mask = [attention_mask, attention_mask, attention_mask]
all_att_mask = torch.stack(all_att_mask, dim=0)

roberta_hidden_states = robert_model(all_sent, attention_mask=all_att_mask)
roberta_last_hidden_states = roberta_hidden_states[0]
states = roberta_last_hidden_states.transpose(1,0)  # (10, 3, 768)
cls_embedding = states[0]  # 一般直接取CLS顶层向量作为整句的语义特征
question_embedding = cls_embedding  # (32, 768) 

print("Done...")