from torch.utils.data import Dataset
import torch
from torch.nn.utils.rnn import pad_sequence

class contrast_poe_dataloader(Dataset):
    def __init__(self, data) -> None:
        self.tweets = [data[i][0] for i in range(len(data))]
        self.y_label = [data[i][1] for i in range(len(data))]
        self.z_label = [data[i][2] for i in range(len(data))]

    def __len__(self):
        return len(self.tweets)
    
    def __getitem__(self, index):
        return self.tweets[index], self.y_label[index], self.z_label[index]

def batch_padding(batch):
    lex, y, z = zip(*batch)

    lex = [torch.tensor(s).cuda() for s in list(lex)]
    y = [torch.tensor(s).cuda() for s in list(y)]
    z = [torch.tensor(s).cuda() for s in list(z)]
    lex = pad_sequence(lex, batch_first=True, padding_value=0)
    y = pad_sequence(y, batch_first=True, padding_value=0)
    z = pad_sequence(z, batch_first=True, padding_value=0)
    output = {}
    mask = lex != 0
    mask_y = y !=0
    mask_z = z !=0
    output["lex"] = lex
    output["label_y"] = y
    output["label_z"] = z
    output["mask"] = mask
    output["mask_y"] = mask_y
    output["mask_z"] = mask_z
    return output

