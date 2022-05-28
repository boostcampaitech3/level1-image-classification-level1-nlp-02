# 나랏말싸미

### 소개

네이버 부스트캠프 AI Tech 나랏말싸미 팀의 "마스크 착용 상태 분류" 대회 Repository입니다.
<br>

### Folder Tree

<br>


```
.
├── config
│   └── config.yaml
├── inference.py
├── models
│   └── model.py
├── readme.md
├── requirements.txt
├── train
│   ├── __init__.py
│   └── trainer.py
├── train.py
└── utils
    ├── change_data.py
    ├── dataloader.py
    ├── metrics.py
    ├── processing.py
    └── utils.py
```

<br>

### Components

**dataset.py**

* 마스크 데이터셋을 읽고 전처리를 진행한 후 데이터를 하나씩 꺼내주는 Dataset 클래스를 구현한 파일입니다.

<br>


**loss.py**

* 이미지 분류에 사용될 수 있는 다양한 Loss 들을 정의한 파일입니다

<br>

**model.py**

* 데이터를 받아 연산을 처리한 후 결과 값을 내는 Model 클래스를 구현하는 파일입니다.

<br>

**train.py**

* 마스크 데이터셋을 통해 CNN 모델 학습을 진행하고 완성된 모델을 저장하는 파일입니다.

```bash
python train.py
```

* 학습에 필요한 하이퍼파라미터는 ./config/config.yaml에서 관리합니다.

<br>

**inference.py**

* 학습 완료된 모델을 통해 test set 에 대한 예측 값을 구하고 이를 .csv 형식으로 저장하는 파일입니다.

```bash
python inference.py
```

<br>


### 프로젝트 주제
 사람의 이미지를 통해 성별, 마스크 착용 상태(제대로 씀, 이상하게 씀, 착용 안함), 그리고 나이(30세 미만, 30세 이상 및 60세 미만, 60세 이상)를 종합하여 총 18개의 클래스를 예측하는 Mask Classification 구축한다.


### 프로젝트 구조 및 사용 데이터셋의 구조도

![모델구조](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/820a85cf-7ea0-49db-9dd8-a60919ccf573/Untitled.png)

모델구조

![데이터셋 구조도](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/d602004b-aa1e-4fa9-beb3-8f967c6f7a5e/Untitled.png)

데이터셋 구조도

- train/images 폴더 내에 인물 별 폴더가 존재하는 구조로 이루어져있다.
- 인물 폴더 내에는 마스크를 안쓰고 있는 이미지 1장, 마스크를 쓰고 있는 이미지 5장, 마스크를 부정확하게 쓰고 있는 이미지 1장으로 구성되어있다.

### 기대효과

  카메라에 비춰진 사람 얼굴 이미지만으로 이 사람이 마스크를 쓰고 있는지, 쓰지 않았는지, 정확히 쓴 것이 맞는지 **자동으로 가려낼 수 있는 시스템**을 구축함으로써, 해당 시스템이 공공장소 입구에 갖춰져 있다면 적은 인적 자원으로도 충분히 검사가 가능할 것이다.

  <br>

## 1-2. 프로젝트 팀 구성 및 역할

송영준 : 모델링, EDA, dataloader 작성

이두호 : 모델을 3개로 분리하여 학습시키는 전략 수행. 협업 툴 설정 및 관리, utils, models 작성

임수정 : 하이퍼파라미터 튜닝, inference code 작성

조혁준 : 데이터 전처리, training code 작성

<br>


## 1-4. 프로젝트 수행 결과

### EDA

총 데이터 : 4,500명 $\times$ 7개의 Mask Case (제대로 씀, 이상하게 씀, 안씀)  =  31,500 장의 이미지

학습 데이터 (총 데이터의 60%) : 18,900 장의 이미지

평가 데이터 (총 데이터의 40%) : 6,300 (public) / 6,300 (private)

![EDA.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/b270b3bc-d823-4889-b775-24c0a301d09d/EDA.png)

- 성별 기준으로 보았을 때, 여성 데이터는 남성 데이터에 비해 619(59.1%)개 만큼 많았다.
- 나이 기준으로 보았을 때, 60세 이상 카테고리 데이터의 수가 30세 미만 카테고리 데이터의 수에 비해 1,089장(85.0%)개 만큼 부족한 데이터 불균형 상태임을 알 수 있다.

<br>

### 데이터 전처리

- 사람단위로 train/valid 나누기 **:**  validation에서 나오는 f1-score(0.9xx)와 실제 test에서의 f1-score(0.6xx)의 차이가 너무 크다는 문제가 나타났다. 데이터를 살펴보니 같은 사람의 이미지가 7장 있기 때문에 사람의 사진이 train과 valid에 나뉘어 들어가는 경우 overfitting이 될 수 있을 것이라 생각하고 사람 단위로 train, valid를 분할하였다.
- 라벨링이 잘못되어있는 데이터를 올바르게 수정하였다.

<br>

### 모델 개요

- EfficientNet-B0 : width, depth, resolution를 조합하는 Compound Scaling을 사용하여 다른 비슷한 성능의 모델보다 적은 파라미터의 수를 가지는 모델이다.
- 입력받은 이미지를 Mask * Gender * Age = 3 * 2 * 3 = 18개의 클래스로 구분한다.
![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/18d44712-c2b6-4ec5-80c7-af342ef8f6b7/Untitled.png)

<br>

### 모델 선정 및 분석

**EfficientNet-B0**

- LB 점수: `0.6867`
- Augmentation
    - `ToGray`
    - `Resize(256X256)`
    - `HorizontalFlip`
    - `Rotate` or `AffineTransform`
    - `RandomBrightnessContrast` or `MotionBlur`
    - `GridDropout` or `Cutout`
    - `Normalize(ImageNet stat)`
- optimizer: `AdamW`  /  LearningRate: `1e-5`
- Loss: `CrossEntropyLoss`
- Epoch: `15`
- Batch-size: `64`

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/e990ef05-0101-4827-a591-e5c6e68b92ba/Untitled.png)

- Gray Scale을 적용하기 이전과 이후를 비교한 학습그래프이다.(녹색이 적용 후)

<br>

**앙상블 방법**

- n-Fold에서 생성된 모델의 예측치를 평균하여 계산하는 방식인 `OOF Ensemble`을 사용했습니다.

<br>

### 자체 평가 의견

- 잘한 점
    - 데일리 스크럼을 통해 현 상황을 적극 공유하였고 이에, 실험이 중복되지 않고 효율적으로 진행되었다.
    - 진행 과정 속 막히는 부분에 대해, 팀원 모두 적극적으로 참여하고 고민해주었다.
- 아쉬웠던 점
    - 노력 대비 성과가 부족하여 아쉬웠다.
    - 각자 다양한 모델로도 실험을 했는데, 각자의 모델을 ensemble 해봤다면 성능이 더 올랐을 수도 있는데, 그 시도를 못해봐서 아쉽다. → 추후 있을 P-Stage에서는 프로젝트 일정 관리를 계획적으로 작성하고 그에 따라 실험을 진행할 것이다.
- 시도했으나 잘 되지 않았던 것들
    - 깃허브를 이용한 협업(이슈, 코드 리뷰)이 원활하게 이루어지지 않았다.
        
        → Commit Log 통일 / Branch 사용 / Pull - Request 를 사용한 코드 리뷰 등을 적극적으로 활용해서 더 완성도 높은 코드를 작성하고 이를 기반으로 효율적인 실험을 할 것이다.
        
    - WandB 를 이용하여 실험을 진행했지만, 프로젝트 단위 관리 및 Artifact 저장소를 제대로 사용하지 못했다. → 다음 P-Stage 전까지 WandB 및 Artifact 사용법을 숙지할 것이다.