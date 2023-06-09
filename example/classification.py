import os
import torch
from torch.utils.data import DataLoader
from server.Wav2vec_model.models.BaseModel import BaseModel
from server.Wav2vec_model.models.wav2vec2 import FeatureExtract, Transcript
from server.Wav2vec_model.utils.dataset_utils import make_batch, TrainSet
# from models.BaseModel import BaseModel
# from models.wav2vec2 import FeatureExtract, Transcript
# from utils.dataset_utils import make_batch, TrainSet

device = "cpu"
module_name = 'classification'
model_name = ['Model1', 'Model2', 'Model3', 'Model4']

model_name = model_name[1]

# 현재 스크립트 파일의 절대 경로를 얻음
script_directory = os.path.dirname(os.path.abspath(__file__))

# 상위 디렉토리 얻기
parent_directory = os.path.dirname(script_directory)

# 경로를 상대 경로로 구성
config_path = os.path.join(parent_directory, 'src/config/clf.config.yaml')
data_path = os.path.join(parent_directory, "src/data/")
ckpt_directory = os.path.join(parent_directory, "src/ckpt/")
pred_directory = os.path.join(parent_directory, "src/test_speech")

classification = BaseModel(config_path, module_name, model_name)
featureExtractor = FeatureExtract().to(device)
transcripter = Transcript().to(device)


def train():
    trainset = TrainSet(data_path, "flac", True)
    train_loader = DataLoader(trainset, batch_size=1)
    classification.fit(train_loader=train_loader,
                       save_path=ckpt_directory, lr=0.0001, epoch_num=5)


def predict():
    input = make_batch(pred_directory, suffle=False).to(device)
    classification.load_ckpt(f"{ckpt_directory}classification-Model2.pt")
    x = featureExtractor(input)
    logits, index = classification.predict(x[:, :128, :])
    select = torch.nonzero(logits > 0.5).squeeze()
    x = x[select]
    index = index[select]
    print(transcripter(x))
    print(index)
