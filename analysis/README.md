# analysis/ ★ 가장 중요

평가 지표 계산과 결과 집계의 단일 진입점.
**모든 V는 여기 정의된 함수를 그대로 호출**한다 — 팀원마다 메트릭 계산이 갈리면 안 됨.

## 구현해야 할 파일

- `metrics.py`
- `aggregate.py`

## 결과 저장 규칙

각 V의 학습이 끝나면 `analysis/v{N}_analysis/{experiment_name}.json` 형태로 결과가 쌓인다.
이 JSON들이 ablation 표의 원천.

```
analysis/
├── v0_analysis/{experiment_name}.json
├── v1_analysis/{experiment_name}.json
├── ...
└── summary.csv  ← aggregate.py가 생성
```

---

## `metrics.py` 명세

### `find_optimal_thresholds(probs, labels, num_classes)`

클래스별로 F1을 최대화하는 임계값을 탐색해 반환.

- **탐색 범위**: `np.arange(0.05, 0.96, 0.01)` (0.05 ~ 0.95, step 0.01)
- 해당 클래스의 양성 샘플이 0이면 → 임계값 `0.5`로 유지하고 스킵
- **반환**: shape `(num_classes,)` float32 numpy 배열

⚠ **이 함수는 best epoch의 validation set에서 한 번만 호출**되어야 한다.
매 epoch마다 호출하면 임계값 자체가 validation에 overfit된다.
최종 평가 때는 best epoch의 임계값을 그대로 재사용 (재계산 금지).

### `compute_metrics(y_true, y_pred_proba, thresholds, class_names)`

5개 지표를 한 번에 계산.

**입력**:
- `y_true`: shape `(N, num_classes)` multi-hot numpy array
- `y_pred_proba`: shape `(N, num_classes)` sigmoid 확률
- `thresholds`: shape `(num_classes,)` — `find_optimal_thresholds`로 얻은 값
- `class_names`: `["N","D","G","C","A","H","M"]`

**반환 dict**:
```python
{
    "macro_auc":      float,                    # sklearn.roc_auc_score(average='macro')
    "per_class_auc":  {class_name: float|None}, # 클래스별 AUC
    "macro_f1":       float,                    # sklearn.f1_score(average='macro', zero_division=0)
    "kappa":          float,                    # per-class quadratic-weighted Kappa의 macro 평균
    "sensitivity":    {class_name: float},      # TP / (TP+FN)
    "specificity":    {class_name: float},      # TN / (TN+FP)
}
```

**세부 규칙**:
- `macro_auc`: 양성/음성이 한쪽뿐인 클래스가 있어 `ValueError`가 발생하면 → `NaN`으로 처리
- `per_class_auc`: `0 < y_true[:, i].sum() < len(y_true)` 인 경우에만 계산, 아니면 `None`
- `macro_f1`: `sklearn.f1_score(..., average='macro', zero_division=0)` 사용
- `kappa`: 클래스별로 `cohen_kappa_score(y_true[:,i], y_pred_binary[:,i], weights='quadratic')`을 구한 뒤 macro 평균.
  양성/음성이 한쪽뿐인 클래스는 제외하고 평균. 리스트가 비면 `0.0`.
- `sensitivity[c]`, `specificity[c]`: threshold 적용 후 이진화한 예측으로 TP/FP/TN/FN 계산

### `save_metrics(metrics, save_dir, experiment_name)`

- `metrics` dict를 JSON으로 저장: `{save_dir}/{experiment_name}.json`
- numpy 타입은 모두 native `float`로 변환 (JSON 직렬화 안전)
- 추가 권장 필드: `timestamp`, `config_snapshot` (실험 config 그대로 사본)

### `evaluate_with_tta(model, loader, device, num_classes, thresholds=None)`

모델 평가 헬퍼. `train/` 코드의 evaluate 로직을 일반화한 버전.

- 각 배치를 원본·좌우 반전 두 번 forward → sigmoid 후 평균
- `thresholds=None`이면:
  → `find_optimal_thresholds`를 호출하여 임계값 계산 후 반환에 포함
- `thresholds`가 주어지면 그대로 사용 (재계산 금지)
- **반환**:
  ```python
  {
      "probs":  np.ndarray,  # (N, num_classes)
      "labels": np.ndarray,  # (N, num_classes)
      "thresholds": np.ndarray,
      **compute_metrics(...)  # 5종 지표 포함
  }
  ```

---

## `aggregate.py` 명세

- `analysis/v*_analysis/*.json`을 **전부** 읽어 ablation 표 생성
- 누락된 실험은 건너뛰고, 존재하는 것만 채움
- **행**: V0 ~ V6 × 모델 3종 (실제 존재하는 것만)
- **열**: `Macro AUC`, `Macro F1`, `Kappa`, 클래스별 AUC (7열), 클래스별 Sens (7열), 클래스별 Spec (7열)
- **출력**:
  - stdout: pandas DataFrame 표
  - 파일: `analysis/summary.csv`

실행:
```bash
python -m analysis.aggregate
```

---

## 결과 폴더 (`v*_analysis/`)

각 V 폴더에는 학습 산출물이 쌓인다:
- `best.pth` — 체크포인트 (gitignore)
- `thresholds.npy` — best 임계값 (gitignore)
- `history.csv` — epoch별 학습 히스토리
- `{experiment_name}.json` — 최종 평가 결과 (★ 이건 git에 커밋 허용)

`.gitkeep` 파일이 들어있어서 빈 폴더도 git에 남는다.
