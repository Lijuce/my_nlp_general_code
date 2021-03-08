# 只有MLM的Albert从0开始预训练，纯transformer框架训练

from transformers import AlbertConfig
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from transformers import AlbertForMaskedLM
from transformers import BertTokenizerFast
from transformers import LineByLineTextDataset
import torch
import os  

os.environ['CUDA_VISIBLE_DEVICES']='0, 1' 

## 分词器训练

files = "./text.txt" # 训练文本文件
vocab_size = 20000
min_frequency = 1
limit_alphabet = 20000
special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] #适用于Bert和Albert

# Initialize a tokenizer
tokenizer = BertWordPieceTokenizer(
    clean_text=True, handle_chinese_chars=True, strip_accents=True, lowercase=True,
)

# Customize training
tokenizer.train(
    files,
    vocab_size = vocab_size,
    min_frequency=min_frequency,
    show_progress=True,
    special_tokens=special_tokens,
    limit_alphabet=limit_alphabet,
    wordpieces_prefix="##"
    )
    
# !mkdir tokenizer
tokenizer.save_model("tokenizer")  # to vocab.txt
tokenizer.save("./tokenizer/tokenizer")

## 模型配置
config = AlbertConfig(
    vocab_size = 20708,
    embedding_size = 256,
    hidden_size = 768,
    num_hidden_layers = 6,
    num_attention_heads = 12,
    intermediate_size = 3072,
    hidden_act = "gelu",
    hidden_dropout_prob = 0.1,
    attention_probs_dropout_prob = 0.1,
    max_position_embeddings = 512,
    type_vocab_size = 2,
    initializer_range = 0.02,
    layer_norm_eps = 1e-12,
)

model = AlbertForMaskedLM(config=config)
# model.num_parameters()  # 查看模型参数量

## 导入训练好的分词器，并生成训练数据
tokenizer = BertTokenizerFast.from_pretrained("./textAlbert/checkpoint-387624", max_len=512) # 分词器导入
dataset = LineByLineTextDataset( # 一行一行的作为一个句子进行MLM
    tokenizer=tokenizer,
    file_path="./text.txt",
    block_size=256,
)

data_collator = DataCollatorForLanguageModeling( # MLM超参数设置
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

training_args = TrainingArguments( # 数据增强设置
    output_dir="./test_albert",
    overwrite_output_dir=True,
    num_train_epochs=2,  # 训练轮次
    per_device_train_batch_size=64,  # batch_size
    save_steps=3126,
    save_total_limit=2
    # prediction_loss_only=True
)
    
trainer = Trainer( # 训练器
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset
)

## 启动训练。训练完会自动保存
trainer.train()  
