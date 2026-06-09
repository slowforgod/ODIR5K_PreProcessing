# data/

학습·평가에 사용하는 모든 데이터 파일이 모이는 폴더.
실제 이미지와 CSV·split JSON은 `.gitignore`로 빠지므로, 팀 공유 링크에서 받아 이 폴더에 둘 것.

## 폴더 구조

```
data/
├── preprocessed_images/        # *.jpg (≈6,392장)
├── full_df.csv                 # 라벨 (이미지 1장 = 1행)
└── patient_split_7class_stratified.json   # 환자 단위 train/val split
```

> ⚠ `splits/` 같은 하위 폴더는 없다. split JSON은 `data/` 바로 아래에 하나로 들어 있다.

---

## 파일 명세

### `preprocessed_images/`
- **JPG** 포맷의 안저 이미지
- 흐름: **crop → resize** 까지 끝난 결과를 미리 만들어 공유한다
  (학습 시 매번 다시 만들지 않음)
- **CLAHE는 적용되어 있지 않다.** 필요한 V(예: V1, V6)에서 학습/평가 시점에 추가로 적용한다.
- 파일명 예: `0_left.jpg`, `0_right.jpg`, `100_left.jpg`, ...
- 파일명은 `full_df.csv`의 `filename` 컬럼과 1:1로 매칭된다 (확장자 포함)

> ⚠ 모든 V(V0~V6)는 이 폴더의 이미지를 입력으로 받는다.
> V0/V2는 CLAHE 없이 그대로 사용, V1/V6는 데이터 로더에서 CLAHE를 1회 적용한다.

---

### `full_df.csv`
**6,392행 (이미지 1장당 1행)**. 컬럼:

| 컬럼 | 의미 |
|------|------|
| `ID` | **환자 ID (정수). split JSON의 patient_id와 매칭되는 키.** |
| `Patient Age`, `Patient Sex` | 메타데이터 |
| `Left-Fundus`, `Right-Fundus` | 한 환자의 좌·우안 파일명 (둘 다 채워져 있음) |
| `Left-Diagnostic Keywords`, `Right-Diagnostic Keywords` | 임상 키워드 원문 |
| `N`, `D`, `G`, `C`, `A`, `H`, `M` | **7개 클래스 멀티핫 라벨 (0/1) — 학습 타깃** |
| `O` | "Other" 클래스. **학습에 사용하지 않음.** |
| `filepath` | 원본 경로 (참고용, 학습에는 사용 안 함) |
| `labels`, `target` | 사전 가공된 라벨 표기. 본 프로젝트에선 N..M 컬럼 7개만 사용. |
| `filename` | **`preprocessed_images/` 안의 jpg 파일명. 데이터 로딩의 기준 키.** |

#### 라벨 규칙 (중요)

- 모델이 예측할 클래스는 **`N, D, G, C, A, H, M` 7종**이다. `O`는 출력에 포함하지 않는다.
- CSV에는 **`N..M`이 모두 0이고 `O`만 1인 행이 894개 존재**한다 (O-only 케이스).
- 이 894행은 **split JSON의 `train_patients`·`val_patients` 양쪽에서 모두 제외**되어 있다.
  → dataset 클래스는 split 멤버십으로 필터링하면 자연스럽게 O-only가 빠진다.
- `N..M`이 전부 0이면서 `O`도 0인 행은 없음 (CSV는 키워드 매칭 1개 이상은 보장).

#### 행 단위 vs 환자 단위

- CSV는 **이미지 단위** (한 행 = 한 jpg)
- 라벨은 환자 단위 진단의 부산물이라 같은 환자의 좌·우안 행이 동일한 N..M 패턴을 갖는 경우가 흔함 (ODIR-5K의 알려진 특성)
- split은 환자 단위(`ID`)로 나뉘므로 좌·우안 leakage는 발생하지 않는다

---

### `patient_split_7class_stratified.json`

환자 단위 train/val split. **하나의 JSON 파일**이며 train·val 양쪽이 모두 들어 있다.

#### 키 구조

```json
{
  "method": "MultilabelStratifiedShuffleSplit (patient-level)",
  "classes": ["N", "D", "G", "C", "A", "H", "M"],
  "o_class": "excluded",
  "o_only_removed": 894,
  "patient_id_col": "ID",
  "train_patients": [0, 2, 4, 5, 6, ...],
  "val_patients":   [1, 18, 27, 32, 37, ...],
  "train_size": 4389,
  "val_size":   1109,
  "total_size": 5498,
  "overlap": 0,
  "val_ratios": {"N": 20.1, "D": 20.0, ...}
}
```

| 필드 | 의미 |
|------|------|
| `train_patients` | 학습용 환자 ID 리스트 (2,280명) |
| `val_patients` | 검증용 환자 ID 리스트 (572명) |
| `train_size`, `val_size` | 해당 patient들에 속하는 **이미지(jpg) 개수** |
| `o_only_removed` | O-only 894 케이스가 split에서 제외되었음을 의미 |
| `patient_id_col` | `"ID"` — CSV에서 매칭에 쓸 컬럼 |
| `val_ratios` | 클래스별 val 비율 (대략 20% 근처에서 stratified) |

#### 분할 방법

- `MultilabelStratifiedShuffleSplit` (다중 라벨 stratified, 환자 단위)
- 7개 클래스 비율을 유지하면서 환자를 분할
- 결과적으로 약 **8:2** (정확히는 4389:1109 ≈ 79.8:20.2)
- O-only 894명은 split 양쪽에서 모두 제외됨

#### dataset에서의 사용 흐름

```python
# 의사 코드
with open('data/patient_split_7class_stratified.json') as f:
    split = json.load(f)

train_ids = set(split['train_patients'])
val_ids   = set(split['val_patients'])

df = pd.read_csv('data/full_df.csv')
train_df = df[df['ID'].isin(train_ids)]
val_df   = df[df['ID'].isin(val_ids)]
# → 이 시점에서 O-only 894행은 자동 제외됨

# 각 행: filename → preprocessed_images/{filename}, 라벨 = [N,D,G,C,A,H,M]
```

---

## 다운로드 안내

팀 Drive에서 받아 다음 구조로 둘 것:

```
data/
├── preprocessed_images/
│   ├── 0_left.jpg
│   ├── 0_right.jpg
│   └── ...
├── full_df.csv
└── patient_split_7class_stratified.json
```

`.gitignore`에 의해 git에는 올라가지 않는다.
