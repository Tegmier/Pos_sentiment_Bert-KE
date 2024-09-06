import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import time
import torch.optim as optim
import numpy as np
from transformers import BertModel, BertTokenizer
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.utils.data import DataLoader
import warnings
from utils.dataloader import KE_Dataloader, batch_padding_tokenizing_collate_function
from model.model_FAN_attention import FinetuneBertFANAttention
from utils.data_import import data_import
import utils.evaluation_tools as evaluation_tools


# 禁用特定类型的警告，例如 FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)
bert_model = BertModel.from_pretrained('bert-base-uncased',
                                    #    quantization_config=bnb_config,
                                       device_map='cuda:0')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Global Training Parameters
embedding_dim = bert_model.config.hidden_size
y_dim = 2
z_dim = 5
batchsize = 20
nepochs = 60
labels2idx = {'O': 0, 'B': 1, 'I': 2, 'E': 3, 'S': 4}
lr = 0.00001
lr_after = 0.00000001
step_epoch = 30
max_grad_norm = 5
numberofdata =30000
world_size = 4  # 使用的 GPU 数量
train_test_rate = 0.7

# 训练函数
def train(data):
    # 创建数据集和采样器
    dataset = KE_Dataloader(data)
    data_loader = DataLoader(
        dataset,
        batch_size=batchsize,
        collate_fn=batch_padding_tokenizing_collate_function
    )
    # 初始化模型
    model = FinetuneBertFANAttention(bert_model=bert_model, y_dim=y_dim, z_dim=z_dim, embedding_dim=embedding_dim).cuda()

    # 初始化损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    # 自适应学习率
    scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=create_lr_lambda(step_epoch=step_epoch, lr_before_change=lr, lr_after_change=lr_after))
    for epoch in range(nepochs):
        print(f"Start epoch {epoch + 1}/{nepochs}")
        model.train()
        t_start = time.time()
        train_loss = []
        for inputs in data_loader:
            y = inputs["label_y"].cuda()
            z = inputs["label_z"].cuda()
            optimizer.zero_grad()
            y_pred, z_pred, attention = model(inputs)
            y_pred = y_pred.reshape(-1, y_dim)
            z_pred = z_pred.reshape(-1, z_dim)
            y = y.reshape(-1)
            z = z.reshape(-1)
            loss_y = criterion(y_pred, y)
            loss_z = criterion(z_pred, z)
            loss = (loss_y + loss_z) / 2
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm, norm_type=2)
            optimizer.step()
            train_loss.append(loss.item())
        avg_loss = sum(train_loss) / len(train_loss)
        print(f"Epoch [{epoch + 1}/{nepochs}], Loss: {avg_loss:.8f}, Time: {time.time() - t_start:.2f}s")
        scheduler.step()
    return model

def eval(model,data):
    test_loss = []
    acc = []
    # 创建数据集和采样器
    testdata = KE_Dataloader(data)
    test_loader = DataLoader(
        testdata,
        batch_size=batchsize,
        collate_fn=batch_padding_tokenizing_collate_function
    )
    # 指标计算准备
    total_z, total_z_pred = [], []
    model.eval()
    criterion = nn.CrossEntropyLoss().cuda()
    sentence_based_acc = []
    with torch.no_grad():
        for inputs in test_loader:
            # y/z:(batch_size, seq)
            y = inputs['label_y'].cuda()
            z = inputs['label_z'].cuda()
            batch_size = y.size(0)
            y_pred, z_pred = model(inputs)
            y_pred = y_pred.reshape(-1, y_dim)
            z_pred = z_pred.reshape(-1, z_dim) # z_pred -> (batch*seq, z_dim)
            y = y.reshape(-1)
            z = z.reshape(-1)
            loss_y = criterion(y_pred, y)
            loss_z = criterion(z_pred, z)
            loss = (loss_y + loss_z) / 2
            test_loss.append(loss.item())
            z_pred = z_pred.argmax(dim=1)           
            total_z.append(np.array(z.cpu()))
            total_z_pred.append(np.array(z_pred.cpu()))
            acc.append((z_pred == z).sum().item()/(z_pred.numel()))
            rows_equal = torch.all(z_pred.reshape(batch_size, -1) == z.reshape(batch_size, -1), dim =1)
            sentence_based_acc.append([rows_equal.sum().item(), batch_size])
    test_loss = np.array(test_loss).mean()
    accuracy = np.array(acc).mean()
    print(f'Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}')
    evaluation_tools.metrics_cal(total_z, total_z_pred)
    sentence_based_acc = np.array(sentence_based_acc)
    sentence_acc = np.sum(sentence_based_acc[:,0])/np.sum(sentence_based_acc[:,1])
    print('Sentence based acc:{:.8f}'.format(sentence_acc))

# 自适应学习率模块
import torch.optim as optim
# 定义一个 lambda 函数来调整学习率
def create_lr_lambda(step_epoch, lr_before_change, lr_after_change):
    def lr_lambda(epoch):
        if epoch < step_epoch:
            return 1
        else:
            return lr_after_change/lr_before_change
    return lr_lambda

# 主函数
if __name__ == "__main__":

    training_data, test_data = data_import(numberofdata=numberofdata, train_test_rate=train_test_rate)
    print(len(training_data))
    print(len(test_data))
    model = train(training_data)
    eval(model, test_data)