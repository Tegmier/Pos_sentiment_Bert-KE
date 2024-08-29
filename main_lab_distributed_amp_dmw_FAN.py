import os
import warnings
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data import DataLoader
from model.model_FAN import FinetuneBertFAN, bert_model
from utils.data_import import data_import
from utils.learning_rate import create_lr_lambda
from utils.evaluation_tools import metrics_cal, keyphrase_acc_cal
from utils.dictributed_env_setup import setup, cleanup
from utils.dataloader import KE_Dataloader, batch_padding_tokenizing_collate_function
from utils.loss_manipulation_and_visualization_utils import main_visualization
from utils.pickle_opt import pickle_read, pickle_write

# 禁用特定类型的警告，例如 FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)

#############################################################################
# Global Training Parameters
embedding_dim = bert_model.config.hidden_size
y_dim = 2
z_dim = 5
batchsize = 10
nepochs = 1
labels2idx = {'O': 0, 'B': 1, 'I': 2, 'E': 3, 'S': 4}
lr = 0.00001
lr_after = 0.000001
step_epoch = 30
max_grad_norm = 5
numberofdata = 15000
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
    model = FinetuneBertFAN(bert_model=bert_model, y_dim=y_dim, z_dim=z_dim, embedding_dim=embedding_dim).cuda(rank)
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)  # 包装模型

    # 初始化损失函数和优化器
    criterion = nn.CrossEntropyLoss().cuda(rank)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    #自适应学习率
    scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=create_lr_lambda(step_epoch=step_epoch, lr_before_change=lr, lr_after_change=lr_after))

    # 初始化 GradScaler
    scaler = GradScaler()
    loss_for_visualization = []
    for epoch in range(nepochs):    
        print(f"Rank {rank}, Start epoch {epoch + 1}/{nepochs}")
        model.train()
        t_start = time.time()
        train_loss = []
        for inputs in data_loader:
            with autocast(device_type='cuda'):
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
                loss = (loss_y + loss_z) / 2
            scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm, norm_type=2)
            scaler.step(optimizer=optimizer)
            train_loss.append(loss.item())
            scaler.update()
        avg_loss = sum(train_loss) / len(train_loss)
        print(f"Rank {rank}, Epoch [{epoch + 1}/{nepochs}], Loss: {avg_loss:.8f}, Time: {time.time() - t_start:.2f}s")
        scheduler.step()
        loss_for_visualization.append(avg_loss)
    pickle_write(loss_for_visualization, f"intermediate_data/rank{rank}_loss.pickle")
    if rank==0:
        model_to_save = model.module
        torch.save(model_to_save.state_dict(), f'checkpoint/model_checkpoint_{batchsize}_{step_epoch}_{nepochs}_{numberofdata}.pth')
    cleanup()

def eval(data):
    model = FinetuneBertFAN(bert_model=bert_model, y_dim=y_dim, z_dim=z_dim, embedding_dim=embedding_dim).cuda()
    model.load_state_dict(torch.load(f'checkpoint/model_checkpoint_{batchsize}_{step_epoch}_{nepochs}_{numberofdata}.pth'))
    test_loss = []
    acc_y, acc_z = [], []
    total_z, total_z_pred = [], []
    total_y, total_y_pred = [], []
    sentence_based_acc_y, sentence_based_acc_z = [], []
    keyphrase_acc_y, keyphrase_acc_z = [], []
    keyphrase_acc_y_pred, keyphrase_acc_z_pred = [], []
    # 创建数据集和采样器
    testdata = KE_Dataloader(data)
    test_loader = DataLoader(
        testdata,
        batch_size=batchsize,
        collate_fn=batch_padding_tokenizing_collate_function
    )
    # 指标计算准备
    model.eval()
    criterion = nn.CrossEntropyLoss().cuda()
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
            # prepare for keyphrase_acc_cal
            keyphrase_acc_y.append(np.array(y.reshape(batch_size, -1).cpu()))
            keyphrase_acc_z.append(np.array(z.reshape(batch_size, -1).cpu()))
            # loss calculation
            loss_y = criterion(y_pred, y)
            loss_z = criterion(z_pred, z)
            loss = (loss_y + loss_z) / 2
            test_loss.append(loss.item())
            # use argmax to format the pred result
            y_pred = y_pred.argmax(dim=1)
            z_pred = z_pred.argmax(dim=1)
            # prepare for keyphrase_acc_cal
            keyphrase_acc_y_pred.append(np.array(y_pred.reshape(batch_size, -1).cpu()))
            keyphrase_acc_z_pred.append(np.array(z_pred.reshape(batch_size, -1).cpu()))
            # save labels and predicts
            total_z.append(np.array(z.cpu()))
            total_z_pred.append(np.array(z_pred.cpu()))
            total_y.append(np.array(y.cpu()))
            total_y_pred.append(np.array(y_pred.cpu()))
            # cal acc_y & acc_z
            acc_y.append((y_pred == y).sum().item()/(y_pred.numel()))
            acc_z.append((z_pred == z).sum().item()/(z_pred.numel()))
            # cal rows_equal_y & rows_equal_z
            rows_equal_y = torch.all(y_pred.reshape(batch_size, -1) == y.reshape(batch_size, -1), dim = 1)
            rows_equal_z = torch.all(z_pred.reshape(batch_size, -1) == z.reshape(batch_size, -1), dim = 1)
            sentence_based_acc_y.append([rows_equal_y.sum().item(), batch_size])
            sentence_based_acc_z.append([rows_equal_z.sum().item(), batch_size])
    test_loss = np.array(test_loss).mean()
    accuracy_y = np.array(acc_y).mean()
    accuracy_z = np.array(acc_z).mean()
    print(f'Test Loss: {test_loss:.4f}, Accuracy Y: {accuracy_y:.8f}, Accuracy Z: {accuracy_z:.8f}')
    metrics_cal(total_y, total_y_pred, type = "y task")
    metrics_cal(total_z, total_z_pred, type = "z task")
    keyphrase_acc_cal(keyphrase_acc_y, keyphrase_acc_y_pred, type = "y task")
    keyphrase_acc_cal(keyphrase_acc_z, keyphrase_acc_z_pred, type = "z task")
    sentence_based_acc_y = np.array(sentence_based_acc_y)
    sentence_acc_y = np.sum(sentence_based_acc_y[:,0])/np.sum(sentence_based_acc_y[:,1])
    print('Sentence based acc for y task:{:.8f}'.format(sentence_acc_y))
    sentence_based_acc_z = np.array(sentence_based_acc_z)
    sentence_acc_z = np.sum(sentence_based_acc_z[:,0])/np.sum(sentence_based_acc_z[:,1])
    print('Sentence based acc for z task:{:.8f}'.format(sentence_acc_z))

# 主函数
if __name__ == "__main__":
    # 设置环境变量（用于分布式环境）
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    training_data, test_data = data_import(numberofdata=numberofdata, train_test_rate=train_test_rate)
    # 使用 mp.spawn 启动多个进程
    mp.spawn(train, args=(world_size, training_data), nprocs=world_size, join=True)
    training_loss = main_visualization(world_size)
    eval(test_data)
    