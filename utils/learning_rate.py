import torch.nn.functional as F
import torch

# 自适应学习率模块
# 定义一个 lambda 函数来调整学习率
def create_lr_lambda(step_epoch, lr_before_change, lr_after_change):
    def lr_lambda(epoch):
        if epoch < step_epoch:
            return 1
        else:
            return lr_after_change/lr_before_change
    return lr_lambda
def softmax_weight(multi_task_learning_weight):
    return F.softmax(torch.stack([multi_task_learning_weight, 1- multi_task_learning_weight]))
