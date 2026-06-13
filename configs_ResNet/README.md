# configs_ResNet/

V별 YAML 설정 파일.

## 만들어야 할 파일

- `v0.yaml`
- `v1.yaml`
- `v2.yaml`
- `v3.yaml`
- `v4.yaml`
- `v5.yaml`
- `v6.yaml`

(한 V에 대해 백본을 바꾸려면 `v1_resnet50.yaml`, `v1_densenet.yaml`처럼 늘려도 됨)

## 공통 필드

```yaml
experiment_name: "v1_resnet50_seed42"
model_name: "resnet50"           # resnet50 / densenet / efficientnet
num_classes: 7
class_names: ["N", "D", "G", "C", "A", "H", "M"]

data:
  image_dir: "data/preprocessed_images"          # *.jpg
  csv_path: "data/full_df.csv"
  split_path: "data/patient_split_7class_stratified.json"
  split_train_key: "train_patients"              # split JSON 안의 키
  split_val_key: "val_patients"
  patient_id_col: "ID"                           # full_df.csv 안의 환자 ID 컬럼
  filename_col: "filename"                       # full_df.csv 안의 이미지 파일명 컬럼
  class_cols: ["N", "D", "G", "C", "A", "H", "M"]  # 라벨 컬럼 (O는 제외)

train:
  batch_size: 32
  num_epochs: 30
  lr: 1.0e-4
  weight_decay: 1.0e-4
  dropout: 0.5
  patience: 8
  seed: 42
  device: "cuda"
  num_workers: 4
  use_amp: true

preprocessing:
  # V별로 다름 — 아래 참고

output_dir: "analysis/v1_analysis"
```

## V별 `preprocessing:` 섹션

### v0.yaml
```yaml
preprocessing: {}
```

### v1.yaml
```yaml
preprocessing:
  clip_limit: 2.0
  tile_grid_size: [8, 8]
```

### v2.yaml
```yaml
preprocessing:
  hflip_p: 0.5
  rotate_limit: 15
  rotate_p: 0.5
  shift_limit: 0.05
  scale_limit: 0.05
  ssr_p: 0.5
  brightness_limit: 0.1
  contrast_limit: 0.1
  bc_p: 0.5
```

### v3.yaml
```yaml
preprocessing:
  mixup_alpha: 0.2
  mixup_layers: [0, 1, 2, 3]
```

### v4.yaml (V1 + V2)
```yaml
preprocessing:
  v1:
    clip_limit: 2.0
    tile_grid_size: [8, 8]
  v2:
    hflip_p: 0.5
    rotate_limit: 15
    # ...
```

### v5.yaml (V1 + V3)
```yaml
preprocessing:
  v1:
    clip_limit: 2.0
    tile_grid_size: [8, 8]
  v3:
    mixup_alpha: 0.2
    mixup_layers: [0, 1, 2, 3]
```

### v6.yaml (V1 + V2 + V3)
```yaml
preprocessing:
  v1:
    clip_limit: 2.0
    tile_grid_size: [8, 8]
  v2:
    hflip_p: 0.5
    rotate_limit: 15
    # ...
  v3:
    mixup_alpha: 0.2
    mixup_layers: [0, 1, 2, 3]
```

## 작성 규칙

- `experiment_name`은 결과 JSON 파일명이 되므로 V/모델/seed를 식별 가능하게 작성
- `output_dir`은 V 번호에 맞게 `analysis/v{N}_analysis/` 로 고정
- `model_name`을 바꿔도 동일 V 안에서 비교 가능 (resnet50 / densenet / efficientnet)
- Mixup이 들어가는 V3/V5/V6은 `model_name: "resnet50"`만 사용
