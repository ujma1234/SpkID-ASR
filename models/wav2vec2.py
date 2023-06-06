from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torch
processor = Wav2Vec2Processor.from_pretrained("src/ckpt/")
model = Wav2Vec2ForCTC.from_pretrained("src/ckpt/")

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