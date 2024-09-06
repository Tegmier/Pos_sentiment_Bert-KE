import math

def dmw_weight_cal_norm(loss1, loss2):
    total = loss1 + loss2
    return loss1/total, loss2/total
def dmw_weight_cal_exp(loss1, loss2):
    loss1 = math.exp(loss1)
    loss2 = math.exp(loss2)
    weight1, weight2 = dmw_weight_cal_norm(loss1, loss2)
    return weight1, weight2

def regular_weight_cal(loss1, loss2):
    return 0.5,0.5