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
from model.model_MFAN_MLP_multiBert import MultiBERT
from utils.data_import_multiBERT import data_import, multi_bert_data_process
from utils.learning_rate import create_lr_lambda, create_lr_lambda_decay_by_10th
from utils.evaluation_tools import metrics_cal, keyphrase_acc_cal
from utils.dictributed_env_setup import setup, cleanup
from utils.dataloader import KE_Dataloader, batch_padding_tokenizing_collate_function
from utils.loss_manipulation_and_visualization_utils import main_visualization
from utils.pickle_opt import pickle_read, pickle_write
from utils.loggings import get_logger, write_log
from utils.dmw_cal import dmw_weight_cal_norm, dmw_weight_cal_exp, regular_weight_cal
from functools import partial
# Load model directly
from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("Twitter/twhin-bert-base")
model = AutoModelForMaskedLM.from_pretrained("Twitter/twhin-bert-base")

# 禁用特定类型的警告，例如 FutureWarning
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

#############################################################################
# Global Training Parameters
y_dim = 2
z_dim = 5
batchsize = 64
nepochs = 30
labels2idx = {'O': 0, 'B': 1, 'I': 2, 'E': 3, 'S': 4}
lr = 0.0001
lr_after = 0.000001
step_epoch = 25
max_grad_norm = 5
numberofdata = 40000
world_size = 4  
train_test_rate = 0.7
decay_rate = 0.8
model_name = "MultiBert_lab_distributed_amp_dmw_MFAN_MLP_maskloss_logger"
#############################################################################

#############################################################################

model_name = "bert_model"

if model_name == "bert_model":
    from transformers import BertModel, BertTokenizer
    selected_model = BertModel.from_pretrained('bert-base-uncased')
    selected_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    selected_model_name = "bert_model"
if model_name == "roberta_model":
    from transformers import RobertaModel, RobertaTokenizer
    selected_model = RobertaModel.from_pretrained('roberta-base')
    selected_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    selected_model_name = "roberta_model"
if model_name == "albert_model":
    from transformers import AlbertModel, AlbertTokenizer
    selected_model = AlbertModel.from_pretrained('albert-base-v2')
    selected_tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
    selected_model_name = "albert_model"
if model_name == "distilbert_model":
    from transformers import DistilBertModel, DistilBertTokenizer
    selected_model = DistilBertModel.from_pretrained('distilbert-base-uncased')
    selected_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    selected_model_name = "distilbert_model"
if model_name == "bertweet_model":
    from transformers import AutoModel, AutoTokenizer
    selected_model = AutoModel.from_pretrained('Twitter/twhin-bert-base')
    selected_tokenizer = AutoTokenizer.from_pretrained('Twitter/twhin-bert-base')
    selected_model_name = "bertweet_model"

embedding_dim = selected_model.config.hidden_size
#############################################################################

# 训练函数
def train(rank, world_size, data, logger):
    # 设置分布式环境
    setup(rank, world_size)
    # 创建数据集和采样器
    dataset = KE_Dataloader(data)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    batch_padding_tokenizing_collate_function_with_params = partial(batch_padding_tokenizing_collate_function, tokenizer = selected_tokenizer)
    data_loader = DataLoader(
        dataset,
        batch_size=batchsize,
        sampler=sampler,
        collate_fn=batch_padding_tokenizing_collate_function_with_params
    )

    # 初始化模型
    model = MultiBERT(bert_model=selected_model, y_dim=y_dim, z_dim=z_dim, embedding_dim=embedding_dim).cuda(rank)
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)  

    # 初始化损失函数和优化器
    criterion = nn.CrossEntropyLoss(reduction='none').cuda(rank)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    #自适应学习率
    # scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=create_lr_lambda(step_epoch=step_epoch, lr_before_change=lr, lr_after_change=lr_after))
    scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=create_lr_lambda_decay_by_10th(step_epoch=step_epoch, decay_rate=decay_rate, lr_before_change=lr))
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
                # attention_mask batch*seq
                attention_mask = inputs["attention_mask"].cuda(rank)
                optimizer.zero_grad()
                y_pred, z_pred = model(inputs)
                y_pred = y_pred.reshape(-1, y_dim)
                z_pred = z_pred.reshape(-1, z_dim)
                y = y.reshape(-1)
                z = z.reshape(-1)
                loss_y = criterion(y_pred, y) * attention_mask.reshape(-1)
                loss_z = criterion(z_pred, z) * attention_mask.reshape(-1)
                loss_y = loss_y.mean()
                loss_z = loss_z.mean()
                weight_task1, weight_task2 = dmw_weight_cal_exp(loss_y, loss_z)
                loss = weight_task1 * loss_y + weight_task2 * loss_z
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
    if rank == 0:
        model_to_save = model.module
        torch.save(model_to_save.state_dict(), f'checkpoint/model_checkpoint_{model_name}_{batchsize}_{step_epoch}_{nepochs}_{numberofdata}.pth')
    cleanup()

def eval(data, logger):
    model = MultiBERT(bert_model=selected_model, y_dim=y_dim, z_dim=z_dim, embedding_dim=embedding_dim).cuda()
    model.load_state_dict(torch.load(f'checkpoint/model_checkpoint_{model_name}_{batchsize}_{step_epoch}_{nepochs}_{numberofdata}.pth'))
    test_loss = []
    acc_y, acc_z = [], []
    total_z, total_z_pred = [], []
    total_y, total_y_pred = [], []
    sentence_based_acc_y, sentence_based_acc_z = [], []
    keyphrase_acc_y, keyphrase_acc_z = [], []
    keyphrase_acc_y_pred, keyphrase_acc_z_pred = [], []
    batch_padding_tokenizing_collate_function_with_params = partial(batch_padding_tokenizing_collate_function, tokenizer = selected_tokenizer)
    # 创建数据集和采样器
    testdata = KE_Dataloader(data)
    test_loader = DataLoader(
        testdata,
        batch_size=batchsize,
        collate_fn=batch_padding_tokenizing_collate_function_with_params
    )
    # 指标计算准备
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="none").cuda()
    with torch.no_grad():
        for inputs in test_loader:
            # keyphrase_acc_cal计算的是keyphrase完整度
            # row_equality计算的是句子完整度
            # y/z:(batch_size, seq)
            y = inputs['label_y'].cuda()
            z = inputs['label_z'].cuda()
            attention_mask = inputs["attention_mask"].cuda()
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
            loss_y = criterion(y_pred, y) * attention_mask.reshape(-1)
            loss_z = criterion(z_pred, z) * attention_mask.reshape(-1)
            loss_y = loss_y.mean()
            loss_z = loss_z.mean()
            weight_task1, weight_task2 = dmw_weight_cal_exp(loss_y, loss_z)
            loss = weight_task1 * loss_y + weight_task2 * loss_z
            test_loss.append(loss.mean().item())
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
            rows_equal_y = torch.all(y_pred.reshape(batch_size, -1)*attention_mask == y.reshape(batch_size, -1)*attention_mask, dim = 1)
            rows_equal_z = torch.all(z_pred.reshape(batch_size, -1)*attention_mask == z.reshape(batch_size, -1)*attention_mask, dim = 1)
            sentence_based_acc_y.append([rows_equal_y.sum().item(), batch_size])
            sentence_based_acc_z.append([rows_equal_z.sum().item(), batch_size])
    test_loss = np.array(test_loss).mean()
    accuracy_y = np.array(acc_y).mean()
    accuracy_z = np.array(acc_z).mean()
    logger.info(f'Test Loss: {test_loss:.4f}, Accuracy Y: {accuracy_y:.8f}, Accuracy Z: {accuracy_z:.8f}')
    metrics_cal(total_y, total_y_pred, type = "y task", logger=logger)
    metrics_cal(total_z, total_z_pred, type = "z task", logger=logger)
    keyphrase_acc_cal(keyphrase_acc_y, keyphrase_acc_y_pred, type = "y task", logger=logger)
    keyphrase_acc_cal(keyphrase_acc_z, keyphrase_acc_z_pred, type = "z task", logger=logger)
    sentence_based_acc_y = np.array(sentence_based_acc_y)
    sentence_acc_y = np.sum(sentence_based_acc_y[:,0])/np.sum(sentence_based_acc_y[:,1])
    logger.info('Sentence based acc for y task:{:.3f}'.format(sentence_acc_y))
    sentence_based_acc_z = np.array(sentence_based_acc_z)
    sentence_acc_z = np.sum(sentence_based_acc_z[:,0])/np.sum(sentence_based_acc_z[:,1])
    logger.info('Sentence based acc for z task:{:.3f}'.format(sentence_acc_z))

# 主函数
if __name__ == "__main__":
    # 设置环境变量（用于分布式环境）
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'

    multi_bert_data_process(selected_tokenizer, selected_model_name)
    training_data, test_data = data_import(numberofdata=numberofdata, train_test_rate=train_test_rate, model_name = selected_model_name)
    print(f"the total number of data is {len(training_data)}")
    # 设置logger
    logger = get_logger(path=f'loggings/model_{model_name}_batchsize{batchsize}_nepochs_{nepochs}_numberofdata_{numberofdata}.txt'
                        , output_to_terminal=True)
    write_log(logger, model_name, embedding_dim, batchsize, nepochs, lr, lr_after, step_epoch, max_grad_norm, numberofdata, world_size, train_test_rate)
    # 使用 mp.spawn 启动多个进程
    mp.spawn(train, args=(world_size, training_data, logger), nprocs=world_size, join=True)
    training_loss = main_visualization(world_size)
    eval(test_data, logger)
    