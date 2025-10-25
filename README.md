# 🏥 Medical Multimodal Report Generation

의료 영상(X-ray, CT, MRI)을 입력받아 자동으로 판독 리포트를 생성하는 멀티모달 AI 시스템

## 📋 프로젝트 개요

### 목표
- 의료 영상에서 자동으로 판독 리포트 생성
- Vision Encoder와 Language Decoder를 결합한 멀티모달 모델 구현
- 의료 도메인에 특화된 정확하고 신뢰할 수 있는 AI 시스템 개발

### 주요 기능
- 의료 영상 자동 분석
- 자연어 리포트 생성
- Attention visualization으로 해석 가능성 제공
- 웹 기반 인터페이스 (Gradio/Streamlit)

## 🏗️ 아키텍처

```
Medical Image Input
        ↓
  Vision Encoder (ResNet/ViT)
        ↓
  Image Features
        ↓
  Cross-Attention Layer ← Text Embeddings
        ↓
  Language Decoder (GPT-2/BART)
        ↓
  Medical Report Output
```

## 🗂️ 프로젝트 구조

```
project1_medical_multimodal/
├── data/                      # 데이터셋
│   ├── raw/                   # 원본 데이터
│   ├── processed/             # 전처리된 데이터
│   └── external/              # 외부 데이터
├── models/                    # 모델 저장소
│   ├── checkpoints/           # 학습 중 체크포인트
│   └── final/                 # 최종 모델
├── notebooks/                 # Jupyter notebooks
│   ├── exploration/           # 데이터 탐색
│   └── experiments/           # 실험 노트북
├── src/                       # 소스 코드
│   ├── data/                  # 데이터 처리
│   │   ├── dataset.py         # Dataset 클래스
│   │   └── preprocessing.py   # 전처리 함수
│   ├── models/                # 모델 정의
│   │   ├── vision_encoder.py  # Vision Encoder
│   │   ├── language_decoder.py # Language Decoder
│   │   └── multimodal_model.py # 통합 모델
│   ├── training/              # 학습 스크립트
│   │   ├── train.py           # 학습 메인
│   │   └── evaluate.py        # 평가
│   ├── inference/             # 추론
│   │   └── predict.py         # 예측
│   └── utils/                 # 유틸리티
│       ├── metrics.py         # 평가 지표
│       └── visualization.py   # 시각화
├── tests/                     # 테스트 코드
├── outputs/                   # 출력물
│   ├── reports/               # 생성된 리포트
│   └── visualizations/        # 시각화 결과
├── configs/                   # 설정 파일
├── docs/                      # 문서
├── requirements.txt           # 패키지 의존성
├── setup.sh                   # 초기 설정 스크립트
├── .gitignore                 # Git 제외 파일
└── README.md                  # 프로젝트 설명
```

## 🚀 시작하기

### 1. 환경 설정

```bash
# 가상환경 생성
python -m venv venv

# 가상환경 활성화 (Windows)
venv\Scripts\activate

# 가상환경 활성화 (Linux/Mac)
source venv/bin/activate

# 패키지 설치
pip install -r requirements.txt
```

### 2. 데이터 준비

```bash
# MIMIC-CXR 데이터셋 다운로드 (PhysioNet 계정 필요)
# https://physionet.org/content/mimic-cxr/2.0.0/

# 데이터 전처리
python src/data/preprocessing.py --input data/raw --output data/processed
```

### 3. 모델 학습

```bash
# 기본 학습
python src/training/train.py --config configs/base_config.yaml

# GPU 사용
python src/training/train.py --config configs/base_config.yaml --gpu 0

# 멀티 GPU 사용
python src/training/train.py --config configs/base_config.yaml --gpus 0,1,2,3
```

### 4. 모델 평가

```bash
python src/training/evaluate.py --model models/final/best_model.pth --data data/processed/test
```

### 5. 추론 및 데모

```bash
# 단일 이미지 예측
python src/inference/predict.py --image path/to/image.jpg --model models/final/best_model.pth

# Gradio 웹 데모 실행
python app.py
```

## 📊 데이터셋

### ROCO (Radiology Objects in COntext) - 주 데이터셋
- **설명**: 의료 영상과 캡션 데이터셋 (X-ray, CT, MRI, 초음파 등)
- **크기**: 
  - Train: ~65,000 이미지-캡션 쌍
  - Validation: ~10,000 쌍
  - Test: ~10,000 쌍
- **링크**: https://github.com/razorx89/roco-dataset
- **데이터 출처**: PubMed Central (https://www.ncbi.nlm.nih.gov/pmc/)
- **특징**: 
  - 다양한 의료 영상 종류 (CT, MRI, X-ray, 초음파 등)
  - PubMed Central에서 수집된 실제 의료 논문의 이미지와 캡션
  - 상대적으로 접근이 쉬움 (공개 데이터셋)
  - 다운로드: Kaggle 또는 직접 수집

## 🎯 Week 1 목표

### Day 1-2: 데이터 준비
- [x] ROCO 데이터셋 다운로드 (Kaggle 또는 GitHub)
- [x] 데이터셋 구조 탐색 및 분석
  - 이미지 종류 분포 (CT, MRI, X-ray 등)
  - 캡션 길이 및 특성 분석
- [x] 데이터 전처리 파이프라인 구축
- [x] Train/Val/Test 분할 확인

### Day 3-4: 베이스라인 모델
- [ ] Vision Encoder 구현 (ResNet50/ViT)
  - Pre-trained weights 로드
  - Feature extraction layer 설정
- [ ] Language Decoder 구현 (GPT-2/BART)
  - Tokenizer 설정
  - 의료 용어 vocabulary 확장 (optional)
- [ ] 기본 학습 루프 작성
  - DataLoader 구현
  - Loss function 정의

### Day 5-7: 멀티모달 융합
- [ ] Cross-attention 메커니즘 구현
  - Image features → Query
  - Text embeddings → Key, Value
- [ ] 이미지-텍스트 융합 레이어
- [ ] 초기 학습 및 검증
  - 소규모 데이터로 overfitting 테스트
  - 전체 데이터셋으로 학습 시작

## 📈 평가 지표

- **BLEU**: 텍스트 유사도
- **ROUGE**: 요약 품질
- **METEOR**: 의미론적 유사도
- **CIDEr**: Consensus-based 평가
- **Clinical Accuracy**: RadGraph 기반

## 🛠️ 기술 스택

- **Deep Learning**: PyTorch, PyTorch Lightning
- **Vision**: torchvision, OpenCV
- **NLP**: Transformers, tokenizers
- **Medical**: pydicom, SimpleITK
- **Visualization**: matplotlib, seaborn, plotly
- **Deployment**: Gradio, Streamlit

## 👤 작성자

- **이름**: 이건희
- **이메일**: dlrjsgml0315@gmail.com
- **GitHub**: https://github.com/Lee-keonhee/

---

**프로젝트 진행 상황**: Week 1 준비 중 🚧

**마지막 업데이트**: 2024-10-25
