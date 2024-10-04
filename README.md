# 문제의 수식 및 텍스트 인식을 위한 Scene Text Detection 시스템

## 개요
CRAFT 모델: text_score 및 link_score를 활용하여 문자(character) 및 단어(word) 단위의 텍스트 검출을 추출할 수 있는 모델.   
모델에 관한 자세한 내용은 [CRAFT 논문](https://arxiv.org/abs/1904.01941)을 참조하세요.   

TricubeNet 모델: Feature Pyramid Network(FPN) 형태의 백본을 사용하여 위치 특징 추출 성능이 향상되었으며, Multi Angle Convolution을 통해 다양한 각도의 텍스트 방향을 효과적으로 처리하며, 단어(word) 단위의 텍스트 검출이 가능한 모델.   
모델에 관한 자세한 내용은 [TricubeNet 논문](https://arxiv.org/abs/2104.11435)을 참조하세요.   


### 주요 기능
- 문제 인식을 위한 텍스트 및 수식 객체 검출 학습
- 다중 GPU 지원을 통한 고속 학습
- 다양한 데이터 증강 기법 적용

## 목차
1. [사용 방법](#사용-방법)
2. [모델 설명](#모델-설명)
3. [데이터셋](#데이터셋)
4. [성능](#성능)

## 사용 방법

### 학습

기본 학습 명령어:
```shell
python traincraft.py
python traintricube.py
```

주요 학습 옵션:
- `--resume`: 학습 중단 시 재개
- `--batch_size`: 배치 크기 설정
- `--input_size`: 이미지 입력 크기 설정
- `--dataset`: 사용할 데이터셋 지정
- `--epochs`: 학습할 에폭 수 설정
- `--lr`: 학습률 설정
- `--save_path`: 모델 저장 경로 설정
- `--weight_decay`: 가중치 감쇠 설정
- `--random_seed`: 랜덤 시드 설정
- `--amp`: 자동 혼합 정밀도 활성화

## 모델 설명

이 프로젝트는 CRAFT 및 Tricubenet 모델 기반으로 학습되었습니다.
### CRAFT 모델 주요 특징 및 학습 설정
- VGG16 백본기반의 U-net구조
- Feature fusion score map 생성 (Character 및 Word 단위 Feature를 모두 생성후 합침)
- Weakly-Supervised Learning: character 단위 Bounding Box 레이블이 없는 경우, character는 모델이 추론하고 word 레이블만 학습
- Strong-Supervised Learning: character 및 word 단위 레이블을 모두 학습
- Line Kernel 제작 및 학습
- 입력 이미지 크기: 1024
- Adam 옵티마이저, WarmupPolyLR 스케줄러
- Batch size: 16
- 50 에폭 학습

### Tricubenet 모델 주요 특징 및 학습 설정
- Hourglass-104 백본
- Heatmap cascade refinement
- Multi Angle Convolution
- Tricube Kernel 학습
- 입력 이미지 크기: 1024
- Adam 옵티마이저, WarmupPolyLR 스케줄러
- Batch size: 4
- 50 에폭 학습

### 데이터 Augmentation
 - Rotation
 - PhotometricDistortion
 - RandomMirror
 - Normalize

## 데이터셋

### 데이터셋 구조
데이터셋의 구조는 다음과 같습니다:
```shell
dataset/
├── 데이터명/
│   ├── train_images/
│   │   ├── image1.png
│   │   ├── image2.png
│   │   └── ...
│   ├── val_images/
│   │   ├── image1.png
│   │   ├── image2.png
│   │   └── ...
│   ├── train_labels.txt
└── └── val_labels.txt
```

### 각 파일 폴더 모듈 특징
0. dataset
- craft : txt파일 포맷
- tricubenet : PASCAL VOC XML 포맷
1. kernel
- Gaussian Kernel
- Line Kernel (문자 간의 상하 부분이 뭉쳐지지 않게 하기 위하여 제작)
2. loader
- craftloader (이미지, character feature, link feature, Feature map에 대한 확신도 4가지 변수 반환)
- tricubeloader (이미지, word feature, 바운딩 박스 사이즈에 따른 가중치, 모든 바운딩 박스 사이즈에 대한 합의 크기 4가지 변수 반환)
3. loss
- maploss (mean average precision loss)
- ohemloss (online hard example mining loss)
- swm_fpem_loss (size weight mask false positive eample mining loss)
4. models
- architectures (모델을 선택)
- basenet (모델 모듈의 집합)
5. output
- 학습시 라벨 결과 출력 이미지들을 모아두는 공간
6. utils
- preprocessing, postprocessing 이미지 처리 공간
7. weight
- training에 대한 weight 저장 공간

### 학습 데이터셋
|데이터 이름|개수|설명|
|:---:|:---:|:---:|
|Generated Dataset Version 1|10000장|한글 및 수식이 혼합된 Dataset|
|Generated Dataset Version 2|10000장|한글 및 수식이 혼합된 Dataset (여백, 작은 숫자, 수식 추가)|
|AIHUB 한국어 글자체 이미지 Dataset|5000장|정형화된 한글이미지 Dataset|
|AIHUB 금융업 특화 문서 OCR Dataset|5000장|비정형화 한글 이미지 Dataset|
|SynthText 한국어 글자체 이미지|10000장|다양한 배경화면에 대한 한글 이미지 Dataset|


### 검증 데이터셋
BenchMark Dataset
|데이터 이름|개수|설명|
|:---:|:---:|:---:|
|Generated Dataset Version 1|2000장|한글 및 수식이 혼합된 Dataset 평가|
|Generated Dataset Version 2|2000장|한글 및 수식이 혼합된 Dataset (여백, 작은 숫자, 수식 추가) 평가|
|AIHUB 한국어 글자체 이미지 Dataset|1000장|정형화된 한글이미지 Dataset 평가|
|AIHUB 금융업 특화 문서 OCR Dataset|1000장|비정형화 한글 이미지 Dataset  평가|
|SynthText 한국어 글자체 이미지|2000장|다양한 배경화면에 대한 한글 이미지 Dataset 평가|


## 성능
검증 데이터셋 Scene Text Detection 주요 성능 지표:   
precision : 92.32%, recall : 77.22%, hmean(F-score) : 84.03%
