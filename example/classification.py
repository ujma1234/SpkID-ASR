import os
import torch
from torch.utils.data import DataLoader
from models.BaseModel import BaseModel
from models.wav2vec2 import FeatureExtract, Transcript
from utils.dataset_utils import make_batch, TrainSet

device = "cuda:0"
module_name = 'classification'
model_name = ['Model1', 'Model2', 'Model3', 'Model4']

model_name = model_name[1]

config_path = 'src/config/clf.config.yaml'

classification = BaseModel(config_path, module_name, model_name)
featureExtractor = FeatureExtract().to(device)
transcripter = Transcript().to(device)


def train():
    trainset = TrainSet("src/data/", "flac", True)
    train_loader = DataLoader(trainset, batch_size=1)
    classification.fit(train_loader=train_loader, save_path="src/ckpt", lr=0.0001, epoch_num=100)

def predict():
    input = make_batch("src/test_speech", suffle=False).to(device)
    classification.load_ckpt("src/ckpt/classification-Model2.pt")
    x = featureExtractor(input)
    logits, index = classification.predict(x[:,:128,:])
    select = torch.nonzero(logits > 0.5).squeeze()
    x = x[select]
    index = index[select]
    print(transcripter(x))
    print(index)