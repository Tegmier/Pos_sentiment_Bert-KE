import os
import warnings
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data import DataLoader
from utils.model import FinetuneBert, bert_model
from utils.data_import import data_import
from utils.learning_rate import create_lr_lambda
from utils.evaluation_tools import metrics_cal
from utils.dictributed_env_setup import setup, cleanup
from utils.dataloader import KE_Dataloader, batch_padding_tokenizing_collate_function

# 禁用特定类型的警告，例如 FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)

#############################################################################
# Global Training Parameters
embedding_dim = bert_model.config.hidden_size
y_dim = 2
z_dim = 5
batchsize = 10
nepochs = 30
labels2idx = {'O': 0, 'B': 1, 'I': 2, 'E': 3, 'S': 4}
lr = 0.00001
lr_after = 0.00000001
step_epoch = 35
max_grad_norm = 5
numberofdata = 10000
world_size = 4  # 使用的 GPU 数量
train_test_rate = 0.7
#############################################################################

# 训练函数
def train(rank, world_size, data):
    # 设置分布式环境
    setup(rank, world_size)
    # 创建数据集和采样器
    dataset = KE_Dataloader(data)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    data_loader = DataLoader(
        dataset,
        batch_size=batchsize,
        sampler=sampler,
        collate_fn=batch_padding_tokenizing_collate_function
    )

    # 初始化模型
    model = FinetuneBert(bert_model=bert_model, y_dim=y_dim, z_dim=z_dim, embedding_dim=embedding_dim).cuda(rank)
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)  # 包装模型

    # 初始化损失函数和优化器
    criterion = nn.CrossEntropyLoss().cuda(rank)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    #自适应学习率
    scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=create_lr_lambda(step_epoch=step_epoch, lr_before_change=lr, lr_after_change=lr_after))
    for epoch in range(nepochs):
        print(f"Rank {rank}, Start epoch {epoch + 1}/{nepochs}")
        model.train()
        t_start = time.time()
        train_loss = []
        for inputs in data_loader:
            y = inputs["label_y"].cuda(rank)
            z = inputs["label_z"].cuda(rank)
            optimizer.zero_grad()
            y_pred, z_pred = model(inputs)
            y_pred = y_pred.reshape(-1, y_dim)
            z_pred = z_pred.reshape(-1, z_dim)
            y = y.reshape(-1)
            z = z.reshape(-1)
            loss_y = criterion(y_pred, y)
            loss_z = criterion(z_pred, z)
            total_loss = loss_y + loss_z
            weight_task1 = loss_y/total_loss
            weight_task2 = loss_z/total_loss
            # print(f"Rank {rank}, loss_y {loss_y}, loss_z {loss_z}")
            loss = weight_task1*loss_y + weight_task2*loss_z
            # print(f"Rank {rank} weight_task1 {weight_task1} weight_task2 {weight_task2}")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm, norm_type=2)
            optimizer.step()
            train_loss.append(loss.item())
        avg_loss = sum(train_loss) / len(train_loss)
        print(f"Rank {rank}, Epoch [{epoch + 1}/{nepochs}], Loss: {avg_loss:.8f}, Time: {time.time() - t_start:.2f}s")
        scheduler.step()
    if rank ==0:
        model_to_save = model.module
        torch.save(model_to_save.state_dict(), f'checkpoint/model_checkpoint_{batchsize}_{step_epoch}_{nepochs}_{numberofdata}.pth')
    cleanup()

def eval(data):
    model = FinetuneBert(bert_model=bert_model, y_dim=y_dim, z_dim=z_dim, embedding_dim=embedding_dim).cuda()
    model.load_state_dict(torch.load(f'checkpoint/model_checkpoint_{batchsize}_{step_epoch}_{nepochs}_{numberofdata}.pth'))
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
    metrics_cal(total_z, total_z_pred)
    sentence_based_acc = np.array(sentence_based_acc)
    sentence_acc = np.sum(sentence_based_acc[:,0])/np.sum(sentence_based_acc[:,1])
    print('Sentence based acc:{:.8f}'.format(sentence_acc))

# 主函数
if __name__ == "__main__":
    # 设置环境变量（用于分布式环境）
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    training_data, test_data = data_import(numberofdata=numberofdata, train_test_rate=train_test_rate)
    # 使用 mp.spawn 启动多个进程
    mp.spawn(train, args=(world_size, training_data), nprocs=world_size, join=True)
    eval(test_data)
    