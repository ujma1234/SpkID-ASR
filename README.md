# Wav2vec_model

## Installation
### https://huggingface.co/facebook/wav2vec2-large-960h-lv60-self/tree/main
### donwload pytorch_model.bin -> src/ckpt/

## Simple Guide
- 화자 등록 : src/config/clf.config.yaml 파일의 Model2 | out_dim 을 화자 수에 맞게 변경
- 학습 : src/data/ 안에 0~n 폴더안에 각각의 화자에 맞는 음성 파일을 넣고 train 실행
- 추론 : src/test_speech 안에 순서에 맞게 음성 파일을 넣고 predict 실행
  
