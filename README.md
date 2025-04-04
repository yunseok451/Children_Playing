# 🧠 어린이 객체 행동 패턴 분류 모델 (MViTv2 기반)

본 프로젝트는 **어린이 보호구역** 등에서 어린이의 다양한 행동을 감지하고 분석하기 위해  
**MViTv2 (Multiscale Vision Transformers v2)** 모델을 활용하여 개발한 **객체 행동 패턴 분류기**입니다.  
Transformer 기반의 강력한 분류 모델을 활용해 실제 스마트 시티, 교통 안전 시스템 등에 적용 가능성을 목표로 합니다.

---

## 📁 프로젝트 개요

- **모델 명칭**: MViT (Multiscale Vision Transformers)
- **모델 버전**: V2–tiny
- **모델 Task**: 어린이 객체의 행동 패턴 분류 (Classification)
- **프레임워크**: PyTorch + MMPretrain
- **데이터 출처**: [AI Hub - 어린이 행동 패턴 영상 데이터](https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=71796)

---

## ✅ 모델 선정 이유

MViTv2는 다양한 크기와 형태의 객체를 정밀하게 인식하고 분류할 수 있는 최신 Transformer 기반 모델입니다.  
특히 다음과 같은 이유로 본 과제에 적합하다고 판단하였습니다:

- MViTv2 모델은 다양한 크기와 형태의 객체를 정확하게 탐지 및 분류할 수 있는 능력을 갖추고 있으며 어린이 객체 행동 패턴 분류의 경우, 크기와 형태가 다양할 수 있기 때문에 MViTv2의 높은 정확도는 어린이 객체의 행동 패턴을 더 잘 분류하는데 도움이 됨. 추가로 필요에 따라 모델을 조정할 수 있으며 빠른 처리속도로 실제 어린이 보호구역에서 신속하고 정확한 어린이 객체의 행동 패턴 분류가 중요함으로 MVitv2 모델을 선정.

---

## 🧠 모델 개발 과정

### 📌 사용 데이터셋

- **출처**: [AI Hub - 어린이 행동 패턴 영상 데이터셋](https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=71796)
- **구성**: 다양한 환경(실내/실외)에서 촬영된 어린이의 행동 영상 및 라벨링 데이터
- **라벨 예시**:
  - 걷기
  - 뛰기
  - 도로 진입
  - 멈춤
  - 주의 산만 등

### 🛠️ 개발 과정

1. **데이터 전처리**  
   - 클래스별 균형 조정, 프레임 단위 이미지 분리  
   - 학습/검증/테스트셋으로 분할

2. **모델 설정 및 학습**  
   - MMPretrain 프레임워크 기반의 MViTv2-tiny 설정  
   - ImageNet 사전 학습된 가중치 기반 파인튜닝 수행

3. **추론 및 평가**  
   - Confusion Matrix, Accuracy 평가  
   - ONNX 변환을 통한 경량화 및 실시간 적용 가능성 테스트

---

## 💻 실행 환경

| 구성 항목 | 환경 |
|-----------|------|
| OS | Ubuntu 20.04 |
| Python | 3.8+ |
| CUDA | 11.x |
| Framework | PyTorch 1.12+, MMPretrain 최신 |
| GPU | NVIDIA RTX 시리즈 권장 |

---

## 📦 설치 및 실행 방법

```bash
# 1. 가상 환경 생성
conda create -n mvitv2_env python=3.8
conda activate mvitv2_env

# 2. 필수 패키지 설치
pip install torch torchvision
pip install openmim
mim install mmpretrain

---

## 🧪 모델 추론 (Docker 기반)

### 1. 모델 이미지 로드
```bash
docker load -i nia_kids.tar
```

### 2. 이미지 로드 확인
```bash
docker images
```

> 출력 예시:
```
REPOSITORY   TAG     IMAGE ID        CREATED         SIZE
nia_kids     latest  1ff083246493    3 minutes ago   19.2GB
```

### 3. 컨테이너 실행 및 추론
```bash
docker run -v 테스트데이터경로:/workspace/mmpretrain/kids/test_data 
-v 결과파일확인경로:/workspace/mmpretrain/test_result -e TZ=Asia/Seoul -p 80:8080 –it --shm-size=8G --gpus all --name 컨테이너이름 이미지이름 
```

### 4. 컨테이너 접속
```bash
root@User명 : /mmpretrain# ---> ls
위 문장이 출력된다면 컨테이너 내부에 접속되었습니다.
```

---

## ⚙️ 추론 명령어 요약

### pretrain 파일 적용
```bash
./entrypoint.sh
overwrite!! 가 뜨면 완료
```

### 데이터 가공
```bash
python kids/data_transform.py
```

### 행동 패턴 예측 실행
```bash
python tools/test.py kids/config.py kids/checkpoint.bin --work-dir tta_test --out result.pkl
```

### CSV 변환
```bash
python result.py result.pkl
```

### 결과 확인 및 복사
```bash
cp -f result.csv test_result/
```

---

## 📊 결과 예시

| 이미지 경로        | 예측 라벨 | 실제 라벨 |
|--------------------|------------|------------|
| kids/test_c/0.png  | 1          | 1          |
| kids/test_c/1.png  | 7          | 7          |
| kids/test_c/2.png  | 7          | 7          |

---

## 🔗 관련 자료

- ✅ [MMPretrain MViT 구현 코드](https://github.com/open-mmlab/mmpretrain/tree/main/configs/mvit)
- 📄 [MViT v1 논문 (arXiv)](https://arxiv.org/abs/2104.11227)
- 📄 [MViT v2 논문 (arXiv)](https://arxiv.org/abs/2112.01526)
- 📂 [AI Hub 데이터셋 상세 보기](https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=71796)

---

## 📈 기대 효과 및 활용 방안

- 어린이 보호구역에서의 **실시간 행동 감지 및 경고 시스템 구축**
- 스마트 CCTV 및 IoT 연계 **지능형 교통 안전 시스템** 개발
- 영상 기반 행위 인식 연구 및 다양한 보행자 행동 분석 응용 가능

---

## 🙋‍♀️ 기여 및 문의

해당 프로젝트는 누구나 자유롭게 활용 및 기여하실 수 있습니다.  
궁금한 점이나 개선 아이디어가 있다면 언제든지 이슈를 등록해주세요!

**담당자**: 최윤석  
📧 nim451@naver.com

---

## 📄 라이선스

본 프로젝트는 [MIT License](./LICENSE)를 따릅니다.
