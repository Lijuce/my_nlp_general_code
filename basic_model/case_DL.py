from transformers import AlbertModel
from transformers import BertTokenizerFast
import torch
import torch.nn as nn
import numpy as np
from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support

# 加载数据集
def read_lines(path_to_file, split_str):
    # 读取普通文本格式的数据：multi->array
    data = []
    try:
        with open(path_to_file, 'r') as f:
            for line in f:
                tmp = [x for x in line.strip().split(split_str)]  # x类型视情况而定
                a, b, c = tmp[0], tmp[1], int(tmp[2])
                a = [int(i) for i in a.split(" ")]
                b = [int(i) for i in b.split(" ")]
                c = [c]
                
                data.append((a,b,c))
    except Exception as e:
        raise e
    print("Reading {}".format(path_to_file))
    return data


data = read_lines("./train.tsv", split_str="\t")

# 训练、测试集切割
train_data = [d for i, d in enumerate(data) if i % 10 != 0]
valid_data = [d for i, d in enumerate(data) if i % 10 == 0]

# 数据生成器
from torch.utils.data import DataLoader, Dataset

class data_generator(Dataset):
    def __init__(self, data, batch_size=64):
        # self.data = iter(data)
        self.data = data
        self.batch_size = batch_size

    def __len__(self):
        return len(self.data)

    def sequence_padding(self, inputs, max_length=None, padding_id=0, mode="post"):
        if max_length is None:
            max_length = 16
        
        pad_width = [(0, 0) for _ in np.shape(inputs)]
        outputs = []
        x = inputs

        x = x[:max_length]
        if mode == "post":
            pad_width[0] = (0, max_length - len(x))
        elif mode == "pre":
            pad_width[0] = (max_length - len(x), 0)
        else:
            raise ValueError('"mode" argument must be "post" or "pre".')
    
        x = np.pad(x, pad_width, 'constant', constant_values=padding_id)
        outputs.append(x)
        return  outputs

    def __getitem__(self, index):
        data_point = self.data[index]
        sent_1 = self.sequence_padding(data_point[0])
        sent_2 = self.sequence_padding(data_point[1])

        sent_1 = torch.tensor(sent_1)
        sent_2 = torch.tensor(sent_2)
        label = torch.tensor(data_point[2])
        return sent_1, sent_2, label 


from tqdm import tqdm

train_gen = data_generator(train_data)
train_generator = DataLoader(train_gen, batch_size=6, shuffle=True)

model = XXX

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = ExponentialLR(optimizer, decay)  # 学习率调整策略
optimizer.zero_grad()
best_score = 0

nb_epochs = 5
if phase == "train":
    for epoch in range(nb_epochs):
        model.train()
        loader = tqdm(train_generator, total=len(train_generator), unit="batches")
        running_loss = 0
        for i_batch, inputs in enumerate(loader):
            model.zero_grad()
            
            loss = model(inputs)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            loader.set_postfix(Loss=running_loss/((i_batch+1)*6), Epoch=epoch)
            loader.set_description('{}/{}'.format(epoch, nb_epochs))
            loader.update()
            pass

        scheduler.step()
        
elif phase == "valid":
    model.eval()
    eps = 0.0001
    answers, score = validate(model=model)
    if score > best_score + eps:
        best_score = score
        no_update = 0
        best_model = model.state_dict()
        print(hops + " hop Validation accuracy (no relation scoring) increased from previous epoch", score)
    elif (score < best_score + eps) and (no_update < patience):
        no_update +=1
        print("Validation accuracy decreases to %f from %f, %d more epoch to check"%(score, best_score, patience-no_update))
    elif no_update == patience:
        print("Model has exceed patience. Saving best model and exiting")
        exit()
    if epoch == nb_epochs-1:
        print("Final Epoch has reached. Stoping and saving model.")
        exit()
    for a in answers:
        with open("result.txt", "a") as f:
            f.writelines(a + '\n')
    with open("result.txt", "a") as f:
        f.writelines("*"*50+'\n')