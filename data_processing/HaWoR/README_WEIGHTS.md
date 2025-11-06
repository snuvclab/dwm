# HaWoR Weights & Data Setup

이 문서는 HaWoR의 weights와 _DATA 파일을 설정하는 방법을 설명합니다.

## 공유 위치

weights와 _DATA 파일은 다음 공유 위치에 저장되어 있습니다:
```
/virtual_lab/jhb_vclab/world_model/HaWoR/
├── weights/
└── _DATA/
```

## 다른 사용자가 파일 복사하기

### 방법 1: 전체 복사 (권장)

```bash
# 자신의 world_model 디렉토리로 이동
cd /path/to/your/world_model

# 공유 위치에서 weights와 _DATA 복사
rsync -av /virtual_lab/jhb_vclab/world_model/HaWoR/ data_processing/HaWoR/
```

### 방법 2: 개별 디렉토리 복사

```bash
# weights만 복사
rsync -av /virtual_lab/jhb_vclab/world_model/HaWoR/weights/ \
    /path/to/your/world_model/data_processing/HaWoR/weights/

# _DATA만 복사
rsync -av /virtual_lab/jhb_vclab/world_model/HaWoR/_DATA/ \
    /path/to/your/world_model/data_processing/HaWoR/_DATA/
```

### 방법 3: 심볼릭 링크 사용 (디스크 공간 절약)

```bash
# weights를 심볼릭 링크로 연결
ln -s /virtual_lab/jhb_vclab/world_model/HaWoR/weights \
    /path/to/your/world_model/data_processing/HaWoR/weights

# _DATA를 심볼릭 링크로 연결
ln -s /virtual_lab/jhb_vclab/world_model/HaWoR/_DATA \
    /path/to/your/world_model/data_processing/HaWoR/_DATA
```

## 파일 업데이트하기 (관리자용)

weights나 _DATA 파일을 업데이트한 후, 공유 위치로 동기화하려면:

```bash
# 프로젝트 루트에서 실행
./sync_hawor_weights.sh
```

또는 수동으로:

```bash
# weights 동기화
rsync -av --delete \
    /virtual_lab/jhb_vclab/byungjun_vclab/world_model/data_processing/HaWoR/weights/ \
    /virtual_lab/jhb_vclab/world_model/HaWoR/weights/

# _DATA 동기화
rsync -av --delete \
    /virtual_lab/jhb_vclab/byungjun_vclab/world_model/data_processing/HaWoR/_DATA/ \
    /virtual_lab/jhb_vclab/world_model/HaWoR/_DATA/
```

## 필요한 파일 확인

다음 파일들이 올바르게 복사되었는지 확인하세요:

### weights/
- `weights/hawor/checkpoints/hawor.ckpt`
- `weights/hawor/checkpoints/infiller.pt`
- `weights/hawor/model_config.yaml`
- `weights/external/detector.pt`
- `weights/external/droid.pth` (선택사항)

### _DATA/
- `_DATA/data/mano/MANO_RIGHT.pkl`
- `_DATA/data_left/mano_left/MANO_LEFT.pkl`

## 문제 해결

### 권한 오류
공유 위치에 접근할 수 없는 경우, 관리자에게 문의하세요.

### 파일이 없는 경우
공유 위치에 파일이 없다면, 관리자가 `sync_hawor_weights.sh`를 실행해야 합니다.

