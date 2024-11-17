# 주식 예측 앱 (Stock Prediction App)

## 소개
LSTM(Long Short-Term Memory) 신경망을 활용하여 60일간의 과거 주가 데이터를 학습하고, 이를 바탕으로 미래 주가를 예측하는 프로그램입니다. 2개의 LSTM 층과 Dropout 층으로 구성된 모델이 주가 데이터 패턴을 학습하여 예측합니다.

## 기능
- 주요 한국 주식 종목의 가격 데이터 로드
- 과거 데이터를 바탕으로 주가 예측
- 예측 결과 그래프 시각화

## 설치 방법
1. 프로젝트를 클론합니다.
   ```bash
   git clone https://github.com/your-repository/stock-prediction
   cd stock-prediction
2. 라이브러리 설치
- bash : pip install yfinance matplotlib pandas tensorflow scikit-learn seaborn tkinter
- cmd : python -m pip install yfinance matplotlib pandas tensorflow scikit-learn seaborn tkinter

## 사용 방법
1. 파일 실행 : python StockPredictionApp.py
2. GUI에서 주식 종목과 예측 기간을 설정하고, 예측 버튼을 눌러 결과를 확인합니다.

## 주의사항
- 데이터는 Yahoo Finance API를 통해 불러오므로 인터넷 연결이 필요합니다.
- LSTM 모델 학습에는 시간이 소요될 수 있으며, 예측 결과는 주가 변동성에 따라 달라질 수 있습니다.
  * 60일치 데이터를 통한 학습 결과로 예측하는 단순 모델입니다.
