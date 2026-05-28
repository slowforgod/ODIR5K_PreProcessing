# ODIR-5K Multi-Label Fundus Image Classification

ODIR-5K 안저 영상 데이터셋을 사용한 다중 라벨 분류 프로젝트.
7가지 전처리 ablation × 3종 백본을 비교한다.

## 프로젝트 개요

- **데이터셋**: ODIR-5K (Ocular Disease Intelligent Recognition, 5,000 patients)
- **태스크**: 다중 라벨 분류 (multi-label classification)
- **클래스 (7개)**: `N`, `D`, `G`, `C`, `A`, `H`, `M`
  - N: Normal
  - D: Diabetes
  - G: Glaucoma
  - C: Cataract
  - A: AMD (Age-related Macular Degeneration)
  - H: Hypertension
  - M: Myopia
  - `Other`는 제외. 키워드 매칭이 하나도 없는 샘플은 학습 자체에서 빠진다.

## 전처리 Ablation (V0 ~ V6)

| 코드 | 설명 |
|------|------|
| V0 | baseline (원본 그대로) |
| V1 | CLAHE (LAB 색공간 L 채널) |
| V2 | data augmentation |
| V3 | Manifold Mixup (feature 레벨) |
| V4 | V1 + V2 |
| V5 | V1 + V3 |
| V6 | V1 + V2 + V3 |

## 백본 (3종)

- ResNet-50
- DenseNet
- EfficientNet

> Manifold Mixup(V3/V5/V6)은 **ResNet-50만 지원**. DenseNet/EfficientNet은 V0/V1/V2/V4에서만 사용.

## 평가 지표 (5종)

1. **Macro AUC** — 클래스별 ROC-AUC의 평균
2. **클래스별 AUC** — 7개 클래스 각각의 ROC-AUC
3. **Macro F1** — 클래스별 F1의 평균 (`zero_division=0`)
4. **Cohen's Kappa** — per-class quadratic-weighted Kappa의 macro 평균
5. **Sensitivity / Specificity** — 클래스별 TP/(TP+FN), TN/(TN+FP)

## 폴더 구조

```
odir5k-project/
├── data/                    # 데이터(이미지·CSV·split) — .gitignore로 빠짐
├── preprocessing/           # V0~V6 전처리 모듈
│   ├── v0/ ... v6/
├── model/                   # 백본 팩토리 (ResNet50/DenseNet/EfficientNet)
├── train/                   # 학습 엔트리 포인트 (V별)
├── analysis/                # 평가 지표, 결과 JSON, 집계
│   ├── v0_analysis/ ... v6_analysis/
├── configs/                 # V별 YAML 설정
├── notebooks/               # Colab 러너 등
├── requirements.txt
└── README.md
```

## 환경 설정

```bash
pip install -r requirements.txt
```

권장 Python: 3.9 이상. CUDA가 있다면 학습 시 AMP가 자동 활성화된다.

## 실행 방법

### 학교 서버 (CUDA)

```bash
# 예: V1 실험
python -m train.train_v1 --config configs/v1.yaml
```

V0 ~ V6 각각에 대해 `train_v{N}.py` 엔트리 포인트를 사용한다.

### Google Colab

`notebooks/colab_runner.ipynb` 참고. 일반적인 흐름:

1. 레포 클론
2. `pip install -r requirements.txt`
3. Google Drive 마운트 → `data/`를 Drive 데이터로 심볼릭 링크
4. `!python -m train.train_v1 --config configs/v1.yaml`
5. `analysis/v{N}_analysis/*.json`을 Drive로 복사

## 팀원별 담당

| 담당 | 모듈 | 담당자 |
|------|------|--------|
| V1 (CLAHE) | `preprocessing/v1/` | TBD |
| V2 (Augmentation) | `preprocessing/v2/` | TBD |
| V3 (Manifold Mixup) | `preprocessing/v3/` + `train/mixup_trainer.py` | TBD |

V0, V4, V5, V6은 V1/V2/V3 모듈을 조합한 wrapper이므로 위 3명이 완성되면 자동으로 동작한다.

## 데이터 준비

`data/README.md`를 먼저 읽을 것. 요약:

1. 팀 공유 링크(Drive 등)에서 받기:
   - 전처리된 이미지: `data/preprocessed_images/*.jpg`
   - 라벨 CSV: `data/full_df.csv` (환자 ID 컬럼명: `ID`, 7 클래스 멀티핫 + `O` 컬럼)
   - 환자 단위 split: `data/patient_split_7class_stratified.json`
     (단일 JSON. 키: `train_patients`, `val_patients`. MultilabelStratifiedShuffleSplit)
2. O-only 894행은 split JSON에서 양쪽 모두 제외됨 → split 멤버십 필터링으로 자동 처리.
3. 모든 데이터 파일은 `.gitignore`에 등록되어 있으니 직접 받아서 배치.

## 결과 집계

각 실험은 `analysis/v{N}_analysis/{experiment_name}.json`에 결과를 남긴다.
모든 V를 돌린 다음:

```bash
python -m analysis.aggregate
```

→ `analysis/summary.csv`로 ablation 표가 출력된다.

## License

MIT — `LICENSE` 참고.
