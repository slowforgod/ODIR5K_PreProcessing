# notebooks/

Jupyter / Colab 노트북.

## 만들어야 할 파일

- `colab_runner.ipynb` — Google Colab에서 학습을 돌리는 러너. 담당자가 추가.

## 권장 셀 구성

### 1. 레포 클론

```python
!git clone https://github.com/<org>/odir5k-project.git
%cd odir5k-project
```

### 2. 의존성 설치

```python
!pip install -r requirements.txt
```

### 3. Google Drive 마운트 + 데이터 심볼릭 링크

```python
from google.colab import drive
drive.mount('/content/drive')

# Drive에 미리 올려둔 데이터를 data/로 링크
!ln -sf /content/drive/MyDrive/ODIR-5K/preprocessed_images               data/preprocessed_images
!ln -sf /content/drive/MyDrive/ODIR-5K/full_df.csv                       data/full_df.csv
!ln -sf /content/drive/MyDrive/ODIR-5K/patient_split_7class_stratified.json  data/patient_split_7class_stratified.json
```

### 4. 학습 실행

```python
!python -m train_ResNet.train_v1 --config configs_ResNet/v1.yaml
```

V를 바꾸려면 `train_v{N}` 과 `configs/v{N}.yaml`을 함께 바꾼다.

### 5. 결과를 Drive로 복사

```python
import shutil
shutil.copytree('analysis/v1_analysis',
                '/content/drive/MyDrive/ODIR-5K/results/v1_analysis',
                dirs_exist_ok=True)
```

## 학교 서버 vs Colab

- 학교 서버: 그대로 `python -m train_ResNet.train_v1 --config configs_ResNet/v1.yaml`
- Colab: 위 셀 구성 그대로
- 동일 코드·동일 config로 동일 결과가 나와야 함 (시드 고정)

## 비고

- 이 폴더는 git에 코드를 두는 곳이지, 실험 산출물 저장소가 아니다.
- 결과 JSON은 `analysis/v{N}_analysis/`로 가야 한다.
