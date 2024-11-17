# 반려동물 탐지 프로그램 (Pet Detection Program)

## 소개
이 프로그램은 객체 탐지를 위한 YOLOv5와 이미지 분류를 위한 CNN(Convolutional Neural Network) 모델을 결합하여 고양이, 개 탐지 프로그램입니다. PyTorch 프레임워크를 기반으로 구현했으며, tkinter로 GUI 인터페이스를 구현했습니다

## 기능
- 사진에서 개와 고양이를 탐지하고 분류
- 탐지된 반려동물 위치와 신뢰도 표시
- GUI를 통해 간편하게 이미지 업로드 및 결과 확인

## 설치 방법
1. 프로젝트 클론
   ```bash
   git clone https://github.com/your-repository/pet-detection
   cd pet-detection

2. 라이브러리 설치
- bash : pip install torch torchvision opencv-python pillow tkinter
- cmd : python -m pip install torch torchvision opencv-python pillow tkinter

## 사용 방법
1. 파일 실행 : python PetDetection.py
2. GUI에서 이미지 파일 선택, 탐지 결과 확인

## 주의사항
- pet_classifier.pth 파일이 프로그램과 동일한 폴더에 있어야 합니다.
- YOLOv5 모델은 초기 로드 시 약간의 시간이 걸릴 수 있습니다.
- COCO dataset을 사용하여 분류했습니다.
