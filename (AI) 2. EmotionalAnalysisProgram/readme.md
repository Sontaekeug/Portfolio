# 감정 분석 프로그램 (Emotional Analysis Program)

## 소개
CNN, LSTM, 어텐션 매커니즘을 결합한 딥러닝 모델을 사용하여 긍정, 중립, 부정의 세가지 감정을 분류, 분석하는 프로그램입니다. PyTorch를 통해 모델을 구현하고, konlpy를 사용해 형태소 분석을 실시했습니다.

## 기능
- 한글 텍스트 감정을 긍정, 중립, 부정으로 분류
- 감정 분석 결과를 GUI에서 제공

## 설치 방법
1. 프로젝트 클론
   ```bash
   git clone https://github.com/your-repository/stock-prediction
   cd stock-prediction
2. 라이브러리 설치
- bash : pip install torch scikit-learn konlpy jsonpickle tkinter
- cmd : python -m pip install torch scikit-learn konlpy jsonpickle tkinter


## 사용 방법
1. 파일 실행 : python python EmotionalAnalysisProgram.py
2. GUI에서 텍스트를 입력하고 감정 분석 결과 확인

## 주의사항
- model.pth 파일이 포함되어야 모델이 정상 작동합니다.
- 텍스트 분석 시 한국어 형태소 분석기 KoNLPy를 사용합니다.
   * 일반적인 채팅을 정확히 분석하려면 지금의 데이터 셋보다 월등히 많은 데이터 셋이 필요합니다.
      (데이터셋 감정별 100,000개 이상 필요.)
