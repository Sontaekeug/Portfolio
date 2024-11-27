import tkinter as tk  # GUI 구성용
from tkinter import ttk, messagebox, font  # Tkinter의 확장 모듈
import pandas as pd  # 데이터 프레임 처리
import numpy as np  # 수학 연산 및 배열 처리
import yfinance as yf  # 금융 데이터 다운로드
import matplotlib.pyplot as plt  # 데이터 시각화
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # Matplotlib을 Tkinter에 포함
from sklearn.preprocessing import MinMaxScaler  # 데이터 정규화
from tensorflow.keras.models import Sequential  # Keras의 순차 모델
from tensorflow.keras.layers import LSTM, Dense, Dropout  # LSTM 및 기타 레이어
from datetime import datetime, timedelta  # 날짜 계산
import seaborn as sns  # 그래프 스타일 조정
import warnings  # 경고 메시지 제어
import matplotlib.font_manager as fm  # 폰트 설정

# 경고 메시지 무시
warnings.filterwarnings('ignore')

# Matplotlib 한글 폰트 설정 (Windows 환경)
plt.rcParams['font.family'] = 'Malgun Gothic'

# 주요 한국 주식 종목과 Yahoo Finance 티커 매핑
STOCK_MAPPING = {
    "삼성전자": "005930.KS",
    "SK하이닉스": "000660.KS",
    "NAVER": "035420.KS",
    "카카오": "035720.KS",
    "현대차": "005380.KS",
    "LG에너지솔루션": "373220.KS",
    "삼성바이오로직스": "207940.KS",
    "삼성SDI": "006400.KS",
    "기아": "000270.KS",
    "현대모비스": "012330.KS",
    "포스코홀딩스": "005490.KS",
    "LG화학": "051910.KS",
    "신한지주": "055550.KS",
    "KB금융": "105560.KS",
    "삼성물산": "028260.KS"
}

# 주식 예측 애플리케이션 클래스
class StockPredictionApp:
    def __init__(self, root):
        # Tkinter 기본 설정
        self.root = root
        self.root.title("주식 예측 프로그램")
        self.root.geometry("1200x800")
        
        # 기본 폰트 설정
        default_font = font.nametofont("TkDefaultFont")
        default_font.configure(family="Malgun Gothic", size=10)
        self.root.option_add("*Font", default_font)
        
        # GUI 구성 요소 생성
        self.create_widgets()
        
        # 데이터와 모델 관련 초기 변수
        self.data = None
        self.model = None
        self.scaler = MinMaxScaler()  # 데이터 정규화를 위한 스케일러
        self.stock_name = None  # 선택된 주식 이름 저장

    def create_widgets(self):
        """GUI 구성 요소 생성"""
        # 입력 영역
        input_frame = ttk.LabelFrame(self.root, text="설정", padding="10")
        input_frame.pack(fill="x", padx=10, pady=5)
        
        # 주식 종목 선택 콤보박스
        ttk.Label(input_frame, text="주식 종목:").grid(row=0, column=0, padx=5)
        self.stock_combo = ttk.Combobox(input_frame, values=list(STOCK_MAPPING.keys()))
        self.stock_combo.grid(row=0, column=1, padx=5)
        self.stock_combo.set("삼성전자")  # 기본값 설정
        
        # 예측 기간 입력 필드
        ttk.Label(input_frame, text="예측 기간(일):").grid(row=0, column=2, padx=5)
        self.days_entry = ttk.Entry(input_frame)
        self.days_entry.grid(row=0, column=3, padx=5)
        self.days_entry.insert(0, "30")  # 기본값: 30일
        
        # 데이터 로드 버튼
        self.load_button = ttk.Button(input_frame, text="데이터 로드", command=self.load_data)
        self.load_button.grid(row=0, column=4, padx=5)
        
        # 예측 실행 버튼
        self.predict_button = ttk.Button(input_frame, text="예측", command=self.predict)
        self.predict_button.grid(row=0, column=5, padx=5)
        
        # 예측 정보 표시 프레임
        info_frame = ttk.LabelFrame(self.root, text="예측 정보", padding="10")
        info_frame.pack(fill="x", padx=10, pady=5)
        
        # 예측 정보 텍스트 레이블
        self.prediction_info = ttk.Label(info_frame, text="")
        self.prediction_info.pack()
        
        # 그래프 표시 프레임
        self.graph_frame = ttk.LabelFrame(self.root, text="그래프", padding="10")
        self.graph_frame.pack(fill="both", expand=True, padx=10, pady=5)

    def load_data(self):
        """주식 데이터 로드"""
        try:
            self.stock_name = self.stock_combo.get()  # 선택된 주식 이름 가져오기
            symbol = STOCK_MAPPING.get(self.stock_name)  # 티커 심볼 조회
            
            if not symbol:
                messagebox.showerror("에러", "올바른 주식 종목을 선택해주세요.")
                return
            
            # Yahoo Finance를 통해 데이터 다운로드
            self.data = yf.download(symbol, start='2020-01-01', end=datetime.now().strftime('%Y-%m-%d'))
            
            if len(self.data) == 0:
                messagebox.showerror("에러", "데이터를 불러올 수 없습니다.")
                return
            
            self.plot_data()  # 데이터 시각화
            messagebox.showinfo("성공", f"{self.stock_name} 데이터 로드 완료")
        
        except Exception as e:
            messagebox.showerror("에러", f"데이터 로드 중 오류 발생: {str(e)}")

    def create_lstm_model(self, input_shape):
        """LSTM 모델 생성"""
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=input_shape),  # LSTM 레이어 (50개 유닛, 시퀀스 출력)
            Dropout(0.2),  # 드롭아웃: 과적합 방지
            LSTM(50, return_sequences=False),  # LSTM 레이어 (마지막 시퀀스만 출력)
            Dropout(0.2),
            Dense(25),  # 완전 연결 레이어
            Dense(1)  # 출력 레이어: 주식 가격 예측
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')  # 모델 컴파일
        return model

    def prepare_data(self, data, lookback=60):
        """LSTM 학습 데이터를 준비"""
        scaled_data = self.scaler.fit_transform(data['Close'].values.reshape(-1, 1))  # 종가 데이터를 정규화
        x, y = [], []
        
        for i in range(lookback, len(scaled_data)):
            x.append(scaled_data[i-lookback:i])  # lookback 기간 만큼 데이터 슬라이싱
            y.append(scaled_data[i])  # 예측할 값
        
        return np.array(x), np.array(y)

    def predict(self):
        """주식 가격 예측"""
        try:
            if self.data is None:
                messagebox.showerror("에러", "먼저 데이터를 로드해주세요.")
                return
            
            # 학습 데이터 준비
            lookback = 60  # LSTM 입력 시퀀스 길이
            x, y = self.prepare_data(self.data, lookback)
            
            # 학습/테스트 데이터 분할
            train_size = int(len(x) * 0.8)  # 학습 데이터 비율
            x_train, x_test = x[:train_size], x[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]
            
            # LSTM 모델 생성 및 학습
            self.model = self.create_lstm_model((lookback, 1))
            self.model.fit(x_train, y_train, batch_size=32, epochs=20, verbose=0)
            
            # 미래 가격 예측
            days = int(self.days_entry.get())
            last_sequence = x_test[-1]  # 마지막 테스트 시퀀스
            predicted_prices = []
            
            for _ in range(days):
                next_pred = self.model.predict(last_sequence.reshape(1, lookback, 1), verbose=0)
                predicted_prices.append(next_pred[0, 0])  # 예측 값 저장
                last_sequence = np.roll(last_sequence, -1)  # 시퀀스를 한 칸 이동
                last_sequence[-1] = next_pred  # 새로운 예측 값 추가
            
            # 예측 값을 원래 단위로 변환
            predicted_prices = self.scaler.inverse_transform(np.array(predicted_prices).reshape(-1, 1))
            
            # 예측 기간 정보 업데이트
            start_date = self.data.index[-1] + timedelta(days=1)
            end_date = start_date + timedelta(days=days)
            self.prediction_info.config(
                text=f"예측 기간: {start_date.strftime('%Y년 %m월 %d일')} ~ {end_date.strftime('%Y년 %m월 %d일')}\n"
                     f"예측 기간: {days}일"
            )
            
            # 예측 결과 시각화
            self.plot_prediction(predicted_prices)
        
        except Exception as e:
            messagebox.showerror("에러", f"예측 중 오류 발생: {str(e)}")

    def plot_data(self):
        """주식 데이터 시각화"""
        for widget in self.graph_frame.winfo_children():
            widget.destroy()  # 기존 그래프 제거
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(self.data.index, self.data['Close'], label='실제 가격')
        ax.set_title(f'{self.stock_name} 주가 추이')
        ax.set_xlabel('날짜')
        ax.set_ylabel('가격 (원)')
        ax.legend()
        ax.grid(True)
        
        # 그래프를 Tkinter에 포함
        canvas = FigureCanvasTkAgg(fig, self.graph_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def plot_prediction(self, predicted_prices):
        """예측 결과 시각화"""
        for widget in self.graph_frame.winfo_children():
            widget.destroy()  # 기존 그래프 제거
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # 실제 데이터 플롯
        ax.plot(self.data.index, self.data['Close'], label='실제 가격')
        
        # 예측 데이터 플롯
        future_dates = pd.date_range(
            start=self.data.index[-1] + timedelta(days=1),
            periods=len(predicted_prices),
            freq='B'  # 영업일 기준
        )
        ax.plot(future_dates, predicted_prices, label='예측 가격', linestyle='--', color='red')
        
        ax.set_title(f'{self.stock_name} 주가 예측')
        ax.set_xlabel('날짜')
        ax.set_ylabel('가격 (원)')
        ax.legend()
        ax.grid(True)
        
        # 그래프를 Tkinter에 포함
        canvas = FigureCanvasTkAgg(fig, self.graph_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

# 프로그램 실행
if __name__ == "__main__":
    root = tk.Tk()
    app = StockPredictionApp(root)
    root.mainloop()
