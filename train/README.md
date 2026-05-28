# train/

학습 루프와 V별 엔트리 포인트.

## 구현해야 할 파일

- `base_trainer.py` — 공통 학습 루프 (V0, V1, V2, V4용)
- `mixup_trainer.py` — `BaseTrainer` 상속, Manifold Mixup 처리 (V3, V5, V6용)
- `train_v0.py` ~ `train_v6.py` — V별 엔트리 포인트

## BaseTrainer 책임

- dataset / dataloader 구성
- 모델 빌드 (`model.base.build_model`)
- 옵티마이저 / 스케줄러 / 손실함수 세팅
- 학습 / 검증 루프
- early stopping, 체크포인트, 메트릭 저장

## 학습 설정 (모든 V 공통)

### 손실
- `nn.BCEWithLogitsLoss(pos_weight=...)`
- `pos_weight[i] = (음성 수) / (양성 수)` — 학습 split에서 계산
- 다중 라벨이므로 sigmoid는 모델 출력에 붙이지 말 것

### 옵티마이저
- `AdamW`
- lr = `1e-4` (권장)
- weight_decay = `1e-4` (권장)

### 스케줄러
- `CosineAnnealingLR(T_max=num_epochs)`

### AMP
- `torch.cuda.amp.autocast` + `GradScaler`
- **CUDA에서만 활성화**. CPU 학습 시에는 비활성화.

### 샘플러 (클래스 불균형 완화)
- `WeightedRandomSampler`
- `sample_weights[i] = (labels[i] * (1 / class_freq)).sum()`
  - `class_freq`: 학습 split의 클래스별 양성 빈도
  - 각 샘플의 양성 클래스에 대한 (1/freq) 합

### TTA (Test-Time Augmentation)
- **검증·테스트 시 좌우 반전 TTA 적용**:
  - 원본 이미지로 한 번 forward → `probs_orig`
  - horizontal flip 후 한 번 더 forward → `probs_flip`
  - 최종 확률 = `(probs_orig + probs_flip) / 2`
- 구현은 `analysis/metrics.py`의 `evaluate_with_tta`를 사용

### Early Stopping
- 기준: **Val Macro AUC**
- `patience = 8`
- best checkpoint와 best 임계값(아래) 저장

### 시드 고정
다음 모두 고정:
- `random.seed(seed)`
- `np.random.seed(seed)`
- `torch.manual_seed(seed)`
- `torch.cuda.manual_seed_all(seed)`
- `torch.backends.cudnn.deterministic = True`
- `torch.backends.cudnn.benchmark = False`

## 학습 히스토리 저장

CSV로 저장 (epoch마다 한 행 append):
```
epoch, train_loss, val_loss, val_macro_auc, val_macro_f1, val_kappa
```
저장 경로: `analysis/v{N}_analysis/history.csv`

## 체크포인트·임계값·결과 저장

| 파일 | 경로 |
|------|------|
| best 체크포인트 | `analysis/v{N}_analysis/best.pth` |
| best 임계값 | `analysis/v{N}_analysis/thresholds.npy` |
| 최종 결과 JSON | `analysis/v{N}_analysis/{experiment_name}.json` |

> **임계값 정책 (중요)**: best epoch의 validation set에서 `find_optimal_thresholds`를
> **한 번만 호출**. 매 epoch마다 호출하면 임계값 자체가 val에 overfit된다.
> 최종 평가 시 이 임계값을 **그대로 재사용** (재계산 금지).

## MixupTrainer 추가 책임 (V3/V5/V6)

`BaseTrainer`를 상속하고 학습 step만 오버라이드.

- 매 epoch마다 `mixup_layers`에서 한 레이어를 무작위 선택
- 모델 forward 호출 시 `y, alpha, mixup_layers` 전달
- 손실:
  ```python
  loss = lam * BCE(logits, label_a) + (1 - lam) * BCE(logits, label_b)
  ```
  또는 `mixed_label = lam*label_a + (1-lam)*label_b`로 한 번에 계산
- 검증은 `BaseTrainer`와 동일 (mixup 없음, TTA는 적용)

## V별 entrypoint

각 `train_v{N}.py`의 표준 구조:

```python
# 의사 코드
def main(args):
    cfg = load_yaml(args.config)
    trainer_cls = MixupTrainer if N in (3, 5, 6) else BaseTrainer
    trainer = trainer_cls(cfg)
    trainer.fit()
    trainer.evaluate_and_save()
```

## 실행

```bash
python -m train.train_v1 --config configs/v1.yaml
```

모든 V에 동일한 패턴.

## 출력 디렉터리

`output_dir`은 config에서 받음. 기본값: `analysis/v{N}_analysis/`.
이 폴더에 `best.pth`, `thresholds.npy`, `history.csv`, `{experiment_name}.json`이 쌓인다.
