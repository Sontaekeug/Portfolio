import tkinter as tk
from tkinter import ttk, messagebox, font
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from datetime import datetime, timedelta
import seaborn as sns
import warnings
import matplotlib.font_manager as fm
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'  # 윈도우의 경우

# 주요 한국 주식 종목 매핑
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

class StockPredictionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("주식 예측 프로그램")
        self.root.geometry("1200x800")
        
        # 기본 폰트 설정
        default_font = font.nametofont("TkDefaultFont")
        default_font.configure(family="Malgun Gothic", size=10)
        self.root.option_add("*Font", default_font)
        
        # GUI 구성요소 초기화
        self.create_widgets()
        
        # 데이터 및 모델 관련 변수
        self.data = None
        self.model = None
        self.scaler = MinMaxScaler()
        self.stock_name = None
        
    def create_widgets(self):
        # 입력 프레임
        input_frame = ttk.LabelFrame(self.root, text="설정", padding="10")
        input_frame.pack(fill="x", padx=10, pady=5)
        
        # 주식 선택 콤보박스
        ttk.Label(input_frame, text="주식 종목:").grid(row=0, column=0, padx=5)
        self.stock_combo = ttk.Combobox(input_frame, values=list(STOCK_MAPPING.keys()))
        self.stock_combo.grid(row=0, column=1, padx=5)
        self.stock_combo.set("삼성전자")  # 기본값
        
        # 예측 기간 입력
        ttk.Label(input_frame, text="예측 기간(일):").grid(row=0, column=2, padx=5)
        self.days_entry = ttk.Entry(input_frame)
        self.days_entry.grid(row=0, column=3, padx=5)
        self.days_entry.insert(0, "30")
        
        # 데이터 로드 버튼
        self.load_button = ttk.Button(input_frame, text="데이터 로드", command=self.load_data)
        self.load_button.grid(row=0, column=4, padx=5)
        
        # 예측 버튼
        self.predict_button = ttk.Button(input_frame, text="예측", command=self.predict)
        self.predict_button.grid(row=0, column=5, padx=5)
        
        # 정보 표시 프레임
        info_frame = ttk.LabelFrame(self.root, text="예측 정보", padding="10")
        info_frame.pack(fill="x", padx=10, pady=5)
        
        # 예측 기간 표시 레이블
        self.prediction_info = ttk.Label(info_frame, text="")
        self.prediction_info.pack()
        
        # 그래프 프레임
        self.graph_frame = ttk.LabelFrame(self.root, text="그래프", padding="10")
        self.graph_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
    def load_data(self):
        try:
            self.stock_name = self.stock_combo.get()
            symbol = STOCK_MAPPING.get(self.stock_name)
            if not symbol:
                messagebox.showerror("에러", "올바른 주식 종목을 선택해주세요.")
                return
                
            # Yahoo Finance에서 데이터 가져오기
            self.data = yf.download(symbol, start='2020-01-01', end=datetime.now().strftime('%Y-%m-%d'))
            
            if len(self.data) == 0:
                messagebox.showerror("에러", "데이터를 불러올 수 없습니다.")
                return
                
            self.plot_data()
            messagebox.showinfo("성공", f"{self.stock_name} 데이터 로드 완료")
            
        except Exception as e:
            messagebox.showerror("에러", f"데이터 로드 중 오류 발생: {str(e)}")
            
    def create_lstm_model(self, input_shape):
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model
        
    def prepare_data(self, data, lookback=60):
        scaled_data = self.scaler.fit_transform(data['Close'].values.reshape(-1, 1))
        x, y = [], []
        
        for i in range(lookback, len(scaled_data)):
            x.append(scaled_data[i-lookback:i])
            y.append(scaled_data[i])
            
        return np.array(x), np.array(y)
        
    def predict(self):
        try:
            if self.data is None:
                messagebox.showerror("에러", "먼저 데이터를 로드해주세요.")
                return
                
            # 데이터 준비
            lookback = 60
            x, y = self.prepare_data(self.data, lookback)
            
            # 학습/테스트 데이터 분할
            train_size = int(len(x) * 0.8)
            x_train, x_test = x[:train_size], x[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]
            
            # 모델 생성 및 학습
            self.model = self.create_lstm_model((lookback, 1))
            self.model.fit(x_train, y_train, batch_size=32, epochs=20, verbose=0)
            
            # 미래 예측
            days = int(self.days_entry.get())
            last_sequence = x_test[-1]
            predicted_prices = []
            
            for _ in range(days):
                next_pred = self.model.predict(last_sequence.reshape(1, lookback, 1), verbose=0)
                predicted_prices.append(next_pred[0, 0])
                last_sequence = np.roll(last_sequence, -1)
                last_sequence[-1] = next_pred
                
            # 예측값 역변환
            predicted_prices = self.scaler.inverse_transform(np.array(predicted_prices).reshape(-1, 1))
            
            # 예측 기간 정보 업데이트
            start_date = self.data.index[-1] + timedelta(days=1)
            end_date = start_date + timedelta(days=days)
            self.prediction_info.config(
                text=f"예측 기간: {start_date.strftime('%Y년 %m월 %d일')} ~ {end_date.strftime('%Y년 %m월 %d일')}\n"
                     f"예측 기간: {days}일"
            )
            
            # 결과 플로팅
            self.plot_prediction(predicted_prices)
            
        except Exception as e:
            messagebox.showerror("에러", f"예측 중 오류 발생: {str(e)}")
            
    def plot_data(self):
        # 기존 그래프 제거
        for widget in self.graph_frame.winfo_children():
            widget.destroy()
            
        # 새로운 그래프 생성
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(self.data.index, self.data['Close'], label='실제 가격')
        ax.set_title(f'{self.stock_name} 주가 추이')
        ax.set_xlabel('날짜')
        ax.set_ylabel('가격 (원)')
        ax.legend()
        ax.grid(True)
        
        # 그래프를 GUI에 추가
        canvas = FigureCanvasTkAgg(fig, self.graph_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        
    def plot_prediction(self, predicted_prices):
        # 기존 그래프 제거
        for widget in self.graph_frame.winfo_children():
            widget.destroy()
            
        # 새로운 그래프 생성
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # 실제 데이터 플로팅
        ax.plot(self.data.index, self.data['Close'], label='실제 가격')
        
        # 예측 데이터 플로팅
        future_dates = pd.date_range(
            start=self.data.index[-1] + timedelta(days=1),
            periods=len(predicted_prices),
            freq='B'
        )
        ax.plot(future_dates, predicted_prices, label='예측 가격', linestyle='--', color='red')
        
        ax.set_title(f'{self.stock_name} 주가 예측')
        ax.set_xlabel('날짜')
        ax.set_ylabel('가격 (원)')
        ax.legend()
        ax.grid(True)
        
        # 그래프를 GUI에 추가
        canvas = FigureCanvasTkAgg(fig, self.graph_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

if __name__ == "__main__":
    root = tk.Tk()
    app = StockPredictionApp(root)
    root.mainloop()
