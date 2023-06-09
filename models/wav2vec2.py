from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torch
import os

# 현재 스크립트 파일의 절대 경로를 얻음
script_directory = os.path.dirname(os.path.abspath(__file__))

# 상위 디렉토리 얻기
parent_directory = os.path.dirname(script_directory)

# 경로를 상대 경로로 구성
ckpt_directory = os.path.join(parent_directory, "src/ckpt/")

# 모델과 프로세서 로드
processor = Wav2Vec2Processor.from_pretrained(ckpt_directory)
model = Wav2Vec2ForCTC.from_pretrained(ckpt_directory)


class FeatureExtract(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = model.wav2vec2.feature_extractor

    def forward(self, x):
        x = self.model(x)
        x = x.transpose(-1, -2)
        return x


class Transcript(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = model.wav2vec2.feature_projection
        self.encoder = model.wav2vec2.encoder
        self.lm_head = model.lm_head

    def forward(self, x):
        x = self.projection(x)[0]
        x = self.encoder(x)[-1]
        x = self.lm_head(x)
        predicted_ids = torch.argmax(x, dim=-1)
        transcription = processor.batch_decode(predicted_ids)
        return transcription
