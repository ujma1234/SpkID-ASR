# Wav2vec_model

## Installation
### https://huggingface.co/facebook/wav2vec2-large-960h-lv60-self/tree/main
### donwload pytorch_model.bin -> src/ckpt/

## Simple Guide
- 화자 등록 : src/config/clf.config.yaml 파일의 Model2 | out_dim 을 화자 수에 맞게 변경
- 학습 : src/data/ 안에 0~n 폴더안에 각각의 화자에 맞는 음성 파일을 넣고 train 실행
- 추론 : src/test_speech 안에 순서에 맞게 음성 파일을 넣고 predict 실행
  

## 모델 생성 및 학습

### 모델 생성

1. src/config/clf.config.yaml 파일 오픈

```jsx
classification:
  Model1: 
    feature_dim: 812
    num_layer: 3
    conv_hidden_dim: 512
    out_din: 115
    dr_rate: 30

  Model2: 
    feature_dim: 512
    conv_hidden_dim: 512
    out_dim: 5 <- modify this number
    dr_rate: 30
	
	...
```

1. 사용한 모델이 Model2를 사용하기 때문에 Model2의 out_dim 을 등록된 화자 수 로 수정

→ config만 수정하고 학습을 돌리면 기존의 checkpoint을 덮어쓰기 때문에 변경된 모델에 맞게 생성됨

⇒ 화자 등록시 반드시 train을 돌려야함

### 모델 학습

1. src/data/ 폴더 안에서 0~화자수 의 폴더명과 화자의 index에 맞는 화자 음성파일을 넣기
    - 음성파일의 format이 flac이 아니라면 example/classification.py 에서 `TrainSet("src/data/", "flac", True)` 의 “flac” 을 지정 format으로 변경합니다. (sample_rate은 반드시 16000 이여야함)
    - 0~화자수 폴더 안의 파일이름은 아무거나 지정해도 상관없음 (테스트를 위해 이름을 맞춰 놓음)
    - 폴더를 변경하고 싶다면 “src/data/”를 지우고 변경하고 싶은 dir 입력 (마지막에 반드시 “/” 필요)
2. [train.py](http://train.py) 실행 
    - 학습 데이터가 적어 epoch을 100으로 지정했는데 시간이 너무 오래걸린다면 lr 혹은 화자 분할 음성 shape을 자유롭게 변경
        - utils/dataset_utils.py → TrainSet class에서 128을 다른 숫자로 변경
        - 해당 숫자로 examples/classification.py → predict() 내부의 logits, index = classification.predict(x[:,:128,:]) 의 128을 변경
        - epoch down
        - train again & inference

## 화자 분류 및 추론

1. src/test_speech 폴더 안에 순서에 맞게 분할된 음성을 넣기
    - 폴더를 바꾸고 싶다면 `input = make_batch("src/test_speech", suffle=False).to(device)` 에서 “src/test_speech” 변경
    - 해당 폴더 안에 들어있는 음성을 순서대로 읽어 배치로 만들어 리턴하는 형식이기 때문에 폴더안에 들어간 음성파일의 이름을 오름차순으로 설정해야함
    - 추론이 끝난 음성은 폴더에서 삭제해야함
2. [predict.py](http://predict.py) 실행
    - 화자 분류 logits의 softmax 확률이 50을 넘지 않는 음성은 미등록 화자로 판단하고 trainscript를 반환하지 않음
    - 폴더 안에있는 모든 음성을 배치로 묶어 추론 진행
    - return값 예시
        - ['IF WE HAD BEEN BROTHER AND SISTER INDEED THERE WAS NOTHING', 'PROFOUND SUFFERING MAKES NOBLE IT SEPARATES ONE OF THE MOST REFINED FORMS OF DISGUISE IS EPICURISM ALONG WITH A CERTAIN OSTENTATIOUS BOLDNESS OF TASTE WHICH TAKES SUFFERING LIGHTLY AND PUTS ITSELF ON THE DEFENSIVE AGAINST ALL THAT IS SORROWFUL AND PROFOUND']
        - tensor([1, 3], device='cuda:0')
     

contact : ujma1234@hanyang.ac.kr
