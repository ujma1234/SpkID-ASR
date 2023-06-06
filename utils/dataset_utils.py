import torch
from torch.utils.data import Dataset
from utils.preprocessing.make_data_list import make_data_list, make_trainlist
from utils.preprocessing.read_audio import with_soundfile
from models.wav2vec2 import FeatureExtract

def make_batch(path, f="flac", suffle=True):
    ds = make_data_list(path, f, suffle)
    data_batch = torch.tensor([], dtype=torch.float32)

    for fname in ds:
        input, _ = with_soundfile(fname[0])
        input = torch.from_numpy(input).float().unsqueeze(0)
        if (input.size()[-1] > 320000):
            input = input[:,:320000]
        else :
            pad = torch.zeros(1, 320000-input.size()[-1])
            input = torch.cat([input, pad], dim=1)
        data_batch = torch.cat([data_batch, input], dim=0)
    return data_batch

class TrainSet(Dataset):
    def __init__(self, path, type="flac", suffle=True):
        self.datalist = make_trainlist(path, type, suffle)
        self.extractor = FeatureExtract()
        self.data = []

        for i in self.datalist:
            input,_ = with_soundfile(i[0])
            input = torch.from_numpy(input).float().unsqueeze(0).to("cuda:0")
            input = self.extractor(input)
            if (input.size()[-2] > 128):
                input = input[:, :128, :]
            else :
                pad = torch.zeros(1, 128-input.size()[-2], 512)
                input = torch.cat([input, pad], dim=1)

            self.data += [input.squeeze(0)]
    def __getitem__(self, index):
        return ([self.data[index]] + [torch.tensor(int(self.datalist[index][1]))])
    
    def __len__(self):
        return (len(self.datalist))