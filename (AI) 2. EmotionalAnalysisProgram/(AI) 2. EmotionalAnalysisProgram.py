import torch  # 딥러닝 프레임워크
import torch.nn as nn  # 신경망 모듈
import torch.nn.functional as F  # 활성화 함수 및 기타 유틸리티
import torch.optim as optim  # 최적화 알고리즘
import numpy as np  # 수학적 연산 및 배열 처리
import tkinter as tk  # GUI 생성
from tkinter import ttk, scrolledtext  # Tkinter GUI 위젯
from datetime import datetime  # 날짜 및 시간 처리
from konlpy.tag import Okt  # 한국어 형태소 분석기
import json  # JSON 데이터 처리
import threading  # 멀티스레딩 지원
import queue  # 멀티스레딩을 위한 큐
import os  # 파일 및 디렉터리 작업
import pickle  # 데이터 직렬화/역직렬화
import re  # 정규 표현식
from sklearn.metrics import precision_recall_fscore_support  # 성능 평가 메트릭

# 감정 분석 모델 클래스
class EmotionAnalysisModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        # 임베딩 레이어 정의
        self.embedding = nn.Embedding(vocab_size, embedding_dim)  # 단어를 벡터로 변환
        self.embedding_dropout = nn.Dropout(0.2)  # 드롭아웃: 과적합 방지
        
        # CNN 레이어: 다양한 필터 크기 적용
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(embedding_dim, hidden_dim, kernel_size=k, padding=k//2)
            for k in [3, 4, 5]  # 필터 크기: 3, 4, 5
        ])
        
        # LSTM 레이어: 양방향 처리
        self.lstm = nn.LSTM(
            hidden_dim * 3,  # CNN 출력 크기 (필터 3개)
            hidden_dim,  # 은닉층 크기
            num_layers=2,  # LSTM 레이어 수
            bidirectional=True,  # 양방향 처리
            dropout=0.3,  # 드롭아웃 적용
            batch_first=True  # 입력 형태: (배치, 시간, 특징)
        )
        
        # 어텐션 메커니즘: 중요한 단어에 가중치 부여
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # 양방향 LSTM 출력 크기
            nn.LeakyReLU(),  # LeakyReLU 활성화 함수
            nn.Linear(hidden_dim, 1),  # 스칼라 가중치 계산
            nn.Softmax(dim=1)  # 가중치 정규화
        )
        
        # 출력 레이어: 최종 감정 예측
        self.dropout = nn.Dropout(0.3)  # 드롭아웃 추가
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)  # 은닉층 연결
        self.leaky_relu = nn.LeakyReLU()  # LeakyReLU 활성화 함수
        self.batch_norm = nn.BatchNorm1d(hidden_dim)  # 배치 정규화
        self.fc2 = nn.Linear(hidden_dim, output_dim)  # 최종 출력 (감정 클래스)

    def forward(self, text):
        # 임베딩 처리
        embedded = self.embedding(text)  # 단어 임베딩
        embedded = self.embedding_dropout(embedded)  # 드롭아웃 적용
        
        # CNN 처리
        conv_input = embedded.transpose(1, 2)  # CNN 입력 형태로 변환
        conv_outputs = []
        for conv in self.conv_layers:
            conv_out = self.leaky_relu(conv(conv_input))  # 활성화 함수 적용
            pooled = F.max_pool1d(conv_out, conv_out.shape[2]).squeeze(2)  # 최대 풀링
            conv_outputs.append(pooled)
        
        conv_output = torch.cat(conv_outputs, dim=1)  # CNN 출력 결합
        conv_output = conv_output.unsqueeze(1).repeat(1, embedded.size(1), 1)  # LSTM 입력 크기 맞춤
        
        # LSTM 처리
        lstm_out, _ = self.lstm(conv_output)
        
        # 어텐션 적용
        attention_weights = self.attention(lstm_out)  # 어텐션 가중치 계산
        attention_output = torch.bmm(lstm_out.transpose(1, 2), attention_weights).squeeze(-1)  # 가중치 기반 출력
        
        # 출력 처리
        output = self.dropout(attention_output)  # 드롭아웃 적용
        output = self.fc1(output)  # 완전 연결층
        output = self.leaky_relu(output)  # 활성화 함수
        output = self.batch_norm(output)  # 배치 정규화
        output = self.dropout(output)  # 드롭아웃 적용
        output = self.fc2(output)  # 최종 출력
        
        return output

# 초기 학습 데이터 로드 함수
def load_initial_training_data():
    """초기 학습 데이터"""
    return [
        # 긍정적인 문장 (20개)
        ("오늘 정말 행복한 하루였어!", 0),
        ("시험 결과가 너무 좋아서 기분이 최고야!", 0),
        ("드디어 프로젝트를 성공적으로 마쳤다!", 0),
        ("새로운 친구를 사귀어서 너무 즐거워", 0),
        ("길에서 우연히 옛 친구를 만나서 반가웠어", 0),
        ("승진 소식을 들어서 정말 기쁘네요", 0),
        ("오랜만에 가족들과 좋은 시간을 보냈어요", 0),
        ("열심히 준비한 발표가 대성공이었어!", 0),
        ("첫 월급을 받아서 너무 설레요", 0),
        ("운동을 시작한 뒤로 몸이 정말 건강해진 것 같아", 0),
        ("새로 산 옷이 너무 마음에 들어", 0),
        ("오늘 요리한 음식이 정말 맛있게 됐어", 0),
        ("여행 계획이 다 짜여서 너무 신나!", 0),
        ("소원하던 일이 이루어져서 감격스러워요", 0),
        ("봉사활동을 하고 나니 마음이 따뜻해졌어요", 0),
        ("좋아하는 가수의 콘서트 티켓을 구했어!", 0),
        ("드디어 자격증 시험에 합격했다!", 0),
        ("새로운 취미를 찾아서 너무 즐거워", 0),
        ("오랜만에 맛있는 식사를 했네요", 0),
        ("예쁜 카페를 발견해서 기분이 좋아요", 0),

        # 부정적인 문장 (20개)
        ("오늘 정말 최악의 하루였어...", 2),
        ("시험을 망쳐서 너무 속상하다", 2),
        ("중요한 약속에 지각해서 미안해...", 2),
        ("친구와 심하게 다퉈서 마음이 아파요", 2),
        ("프로젝트 마감일에 맞추지 못했어", 2),
        ("실수로 중요한 파일을 삭제했어...", 2),
        ("몸이 아파서 병원에 가야할 것 같아", 2),
        ("면접에서 떨어져서 너무 실망스러워", 2),
        ("비가 와서 계획이 다 망쳐졌어", 2),
        ("지갑을 잃어버려서 정말 속상해", 2),
        ("스트레스가 너무 심해서 힘들어요", 2),
        ("친한 친구가 이사를 가서 슬퍼요", 2),
        ("실수로 폰을 떨어뜨려서 화면이 깨졌어", 2),
        ("버스를 놓쳐서 회의에 늦었어요", 2),
        ("좋아하는 음식점이 문을 닫았어요", 2),
        ("중요한 발표를 망쳐서 자책감이 들어요", 2),
        ("애인과 헤어져서 너무 힘들어요", 2),
        ("집에서 키우던 반려동물이 아파요", 2),
        ("월급이 생각보다 너무 적어서 실망했어요", 2),
        ("팀 프로젝트에서 무임승차하는 동료 때문에 스트레스 받아요", 2),

        # 중립적인 문장 (20개)
        ("오늘 날씨가 흐린것 같아요", 1),
        ("이번 주말에는 집에서 쉴 예정이에요", 1),
        ("오늘 점심은 평소처럼 회사 근처 식당에서 먹었어요", 1),
        ("새로운 프로젝트가 시작되었습니다", 1),
        ("버스를 타고 출근했어요", 1),
        ("내일 회의가 있다고 합니다", 1),
        ("커피를 마시면서 일하고 있어요", 1),
        ("이번 달 회의는 온라인으로 진행됩니다", 1),
        ("새로운 동료가 입사했다고 해요", 1),
        ("다음 주에 출장이 예정되어 있습니다", 1),
        ("오늘 업무 보고서를 작성했어요", 1),
        ("내일은 평소보다 일찍 출근해야 해요", 1),
        ("주말에 장을 보러 마트에 갈 예정이에요", 1),
        ("오늘 저녁에는 집에서 요리를 해먹을 거예요", 1),
        ("이번 주는 평소와 비슷하게 지나갔네요", 1),
        ("새로운 업무 시스템을 배우고 있습니다", 1),
        ("오후에 팀 미팅이 있습니다", 1),
        ("이메일을 확인하고 답장을 보냈어요", 1),
        ("내일 일정을 확인하고 있습니다", 1),
        ("프린터로 자료를 출력했습니다", 1)
    ]

# 감정 분석 프로그램의 핵심 클래스
class DeepEmotionAnalyzer:
    def __init__(self):
        # 데이터 저장 경로 설정
        self.data_dir = "emotion_analyzer_data"  # 데이터 저장 디렉터리
        self.vocab_path = os.path.join(self.data_dir, "vocab.json")  # 어휘 사전 경로
        self.model_path = os.path.join(self.data_dir, "model.pth")  # 모델 파일 경로
        self.training_data_path = os.path.join(self.data_dir, "training_data.pkl")  # 학습 데이터 경로
        
        # 감정 키워드 사전 초기화
        self.emotion_keywords = {
            'positive': ['좋아', '행복', '기쁘', '감사', '축하', '사랑', '즐겁', '신나', '훌륭', '멋지'],
            'negative': ['슬프', '힘들', '아프', '싫어', '미안', '속상', '화나', '괴롭', '실망', '후회'],
            'neutral': ['보통', '평범', '일반', '무난', '중간', '그저', '담담']
        }
        
        # 데이터 디렉터리 생성
        os.makedirs(self.data_dir, exist_ok=True)
        
        # 모델 하이퍼파라미터 설정
        self.vocab_size = 8000  # 어휘 크기
        self.embedding_dim = 200  # 임베딩 차원 수
        self.hidden_dim = 128  # 은닉층 차원 수
        self.output_dim = 3  # 출력 차원 수 (긍정, 중립, 부정)
        
        self.okt = Okt()  # 형태소 분석기 초기화
        
        # 어휘 사전 및 학습 데이터 로드
        self.load_vocab()
        self.load_training_data()
        
        # 모델 초기화
        self.initialize_model()
        
        # GUI 초기화
        self.setup_gui()

    def load_vocab(self):
        """어휘 사전을 로드하거나 새로 생성"""
        try:
            if os.path.exists(self.vocab_path):
                with open(self.vocab_path, 'r', encoding='utf-8') as f:
                    self.word_to_idx = json.load(f)  # 단어 -> 인덱스 매핑
                    self.idx_to_word = {v: k for k, v in self.word_to_idx.items()}  # 인덱스 -> 단어 매핑
                    print("어휘 사전을 로드했습니다.")
            else:
                # 어휘 사전 초기화
                self.word_to_idx = {'<PAD>': 0, '<UNK>': 1}
                self.idx_to_word = {0: '<PAD>', 1: '<UNK>'}
                print("새로운 어휘 사전을 생성했습니다.")
        except Exception as e:
            print(f"어휘 사전 로드 중 오류: {e}")
            self.word_to_idx = {'<PAD>': 0, '<UNK>': 1}
            self.idx_to_word = {0: '<PAD>', 1: '<UNK>'}

    def save_vocab(self):
        """어휘 사전을 파일로 저장"""
        try:
            with open(self.vocab_path, 'w', encoding='utf-8') as f:
                json.dump(self.word_to_idx, f, ensure_ascii=False, indent=2)
                print("어휘 사전을 저장했습니다.")
        except Exception as e:
            print(f"어휘 사전 저장 중 오류: {e}")

    def load_training_data(self):
        """학습 데이터를 로드하거나 초기화"""
        try:
            if os.path.exists(self.training_data_path):
                with open(self.training_data_path, 'rb') as f:
                    self.train_data = pickle.load(f)
                    print(f"학습 데이터 {len(self.train_data)}개를 로드했습니다.")
            else:
                # 학습 데이터 초기화
                self.train_data = []
                self.train_data.extend(load_initial_training_data())  # 초기 데이터 추가
                self.save_training_data()
                print(f"초기 학습 데이터 {len(self.train_data)}개를 생성했습니다.")
        except Exception as e:
            print(f"학습 데이터 로드 중 오류: {e}")
            self.train_data = []

    def save_training_data(self):
        """학습 데이터를 파일로 저장"""
        try:
            with open(self.training_data_path, 'wb') as f:
                pickle.dump(self.train_data, f)
                print("학습 데이터를 저장했습니다.")
        except Exception as e:
            print(f"학습 데이터 저장 중 오류: {e}")

    def initialize_model(self):
        """모델 초기화 또는 기존 모델 로드"""
        try:
            if os.path.exists(self.model_path):
                # 저장된 모델 로드
                checkpoint = torch.load(self.model_path)
                self.model = EmotionAnalysisModel(
                    checkpoint['vocab_size'],
                    checkpoint['embedding_dim'],
                    checkpoint['hidden_dim'],
                    checkpoint['output_dim']
                )
                self.model.load_state_dict(checkpoint['model_state_dict'])
                print("기존 모델을 로드했습니다.")
            else:
                # 새 모델 초기화
                self.model = EmotionAnalysisModel(
                    self.vocab_size,
                    self.embedding_dim,
                    self.hidden_dim,
                    self.output_dim
                )
                print("새로운 모델을 초기화했습니다.")
                
            # 모델을 평가 모드로 설정
            self.model.eval()
            
            # GPU 사용 여부 확인
            if torch.cuda.is_available():
                self.model = self.model.cuda()  # 모델을 GPU로 이동
                print("GPU를 사용합니다.")
            else:
                print("CPU를 사용합니다.")
                
        except Exception as e:
            print(f"모델 초기화 중 오류 발생: {e}")
            # 오류 발생 시 새 모델 생성
            self.model = EmotionAnalysisModel(
                self.vocab_size,
                self.embedding_dim,
                self.hidden_dim,
                self.output_dim
            )
            print("오류로 인해 새 모델을 생성했습니다.")

    def get_device(self):
        """사용 가능한 디바이스 반환 (GPU 또는 CPU)"""
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def move_to_device(self, tensor):
        """텐서를 디바이스(GPU/CPU)로 이동"""
        device = self.get_device()
        return tensor.to(device)

    def improved_preprocess_text(self, text):
        """텍스트 전처리 및 어휘 사전 업데이트"""
        # 텍스트 정규화 (소문자 변환 및 특수 문자 제거)
        text = text.lower()
        text = re.sub(r'[^가-힣a-z\s]', ' ', text)
        
        # 형태소 분석
        morphs = self.okt.morphs(text)
        
        # 형태소에 대한 가중치 추가
        weighted_morphs = []
        for morph in morphs:
            weight = 1.0  # 기본 가중치
            
            # 감정 키워드에 따라 가중치 조정
            if morph in self.emotion_keywords['positive']:
                weight = 1.5
            elif morph in self.emotion_keywords['negative']:
                weight = 1.5
            elif morph in self.emotion_keywords['neutral']:
                weight = 1.2
            
            # 단어가 어휘 사전에 없으면 추가
            if morph not in self.word_to_idx and len(self.word_to_idx) < self.vocab_size:
                self.word_to_idx[morph] = len(self.word_to_idx)
                self.idx_to_word[self.word_to_idx[morph]] = morph
            
            # 어휘 사전에서 단어 인덱스를 가져옴
            idx = self.word_to_idx.get(morph, self.word_to_idx['<UNK>'])
            weighted_morphs.extend([idx] * int(weight))
        
        # 텐서를 고정된 길이로 패딩
        max_length = 100  # 최대 시퀀스 길이
        if len(weighted_morphs) < max_length:
            weighted_morphs = weighted_morphs + [0] * (max_length - len(weighted_morphs))
        else:
            weighted_morphs = weighted_morphs[:max_length]
        
        return torch.tensor(weighted_morphs).unsqueeze(0)  # 배치 차원 추가

    def train_model(self):
        """모델 학습을 수행"""
        if len(self.train_data) < 5:
            # 학습 데이터가 부족하면 알림
            self.status_var.set("최소 5개의 학습 예제가 필요합니다.")
            return
        
        self.status_var.set("학습 중...")  # 상태 메시지 업데이트
        self.window.update()  # GUI 업데이트
        
        try:
            # 데이터 준비
            X = []
            y = []
            for text, label in self.train_data:
                input_tensor = self.improved_preprocess_text(text)  # 텍스트 전처리
                X.append(input_tensor)
                y.append(label)
            
            X = torch.cat(X)  # 텐서로 변환
            y = torch.tensor(y)  # 레이블 텐서 생성
            
            # 데이터셋 생성
            dataset = torch.utils.data.TensorDataset(X, y)
            batch_size = min(32, len(dataset))  # 배치 크기 설정
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True  # 데이터 섞기
            )
            
            # 클래스 가중치 계산
            class_counts = torch.bincount(y, minlength=3)  # 클래스별 개수 계산
            weights = 1.0 / class_counts.float()  # 가중치 계산
            weights = weights / weights.sum()  # 정규화
            criterion = nn.CrossEntropyLoss(weight=weights)  # 가중치 기반 손실 함수
            
            # 옵티마이저 설정
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=0.001,  # 학습률
                weight_decay=0.001  # 가중치 감쇠
            )
            
            # 학습
            for epoch in range(20):  # 에폭 수
                self.model.train()  # 모델 학습 모드 전환
                total_loss = 0
                
                for batch_X, batch_y in dataloader:
                    optimizer.zero_grad()  # 그래디언트 초기화
                    outputs = self.model(batch_X)  # 모델 예측
                    loss = criterion(outputs, batch_y)  # 손실 계산
                    loss.backward()  # 역전파
                    optimizer.step()  # 매개변수 업데이트
                    total_loss += loss.item()  # 손실 누적
                
                # 상태 업데이트
                status_msg = f"Epoch {epoch + 1}/20 | Loss: {total_loss / len(dataloader):.4f}"
                self.status_var.set(status_msg)
                self.window.update()
            
            self.save_model()  # 학습된 모델 저장
            self.status_var.set("학습 완료!")
        
        except Exception as e:
            self.status_var.set(f"학습 중 오류 발생: {str(e)}")
            print(f"상세 오류: {e}")

    def analyze_emotion(self, text):
        """입력 텍스트의 감정을 분석"""
        try:
            input_tensor = self.improved_preprocess_text(text)  # 입력 텍스트 전처리
            
            with torch.no_grad():
                self.model.eval()  # 모델 평가 모드
                outputs = self.model(input_tensor)  # 감정 예측
                probabilities = torch.softmax(outputs, dim=1)  # 확률 계산
                
                # 감정별 확률
                positive_prob = probabilities[0, 0].item()
                neutral_prob = probabilities[0, 1].item()
                negative_prob = probabilities[0, 2].item()
                
                # 가장 높은 확률의 감정 선택
                probs = [positive_prob, neutral_prob, negative_prob]
                labels = ["긍정 😊", "중립 😐", "부정 😢"]
                max_index = np.argmax(probs)
                confidence = probs[max_index]
                emotion = labels[max_index]
                
                # 신뢰도에 따라 레벨 표시
                confidence_level = (
                    "높음" if confidence > 0.6 else "중간" if confidence > 0.4 else "낮음"
                )
                
                return f"{emotion} (신뢰도: {confidence_level}, 긍정: {positive_prob:.2f}, 중립: {neutral_prob:.2f}, 부정: {negative_prob:.2f})"
        
        except Exception as e:
            return f"감정 분석 중 오류 발생: {str(e)}"

    def setup_gui(self):
        """GUI 설정"""
        self.window = tk.Tk()  # Tkinter 윈도우 생성
        self.window.title("감정 분석 프로그램")  # 창 제목
        self.window.geometry("500x600")  # 창 크기 설정
        
        # 메인 프레임 설정
        main_frame = ttk.Frame(self.window)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 채팅 영역 설정
        self.chat_area = scrolledtext.ScrolledText(
            main_frame, wrap=tk.WORD, font=("맑은 고딕", 12), height=20
        )
        self.chat_area.pack(fill=tk.BOTH, expand=True)
        self.chat_area.config(state=tk.DISABLED)  # 편집 불가
        
        # 입력 영역 설정
        input_frame = ttk.Frame(main_frame)
        input_frame.pack(fill=tk.X, pady=5)
        
        self.input_field = ttk.Entry(input_frame, font=("맑은 고딕", 12))
        self.input_field.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        self.input_field.bind("<Return>", lambda e: self.send_message())  # 엔터 키 이벤트
        
        send_button = ttk.Button(input_frame, text="전송", command=self.send_message)
        send_button.pack(side=tk.RIGHT)
        
        # 상태 표시 라벨
        self.status_var = tk.StringVar(value="상태: 준비됨")
        status_label = ttk.Label(main_frame, textvariable=self.status_var, anchor=tk.W)
        status_label.pack(fill=tk.X, pady=5)

    def send_message(self):
        """사용자 메시지 전송 및 감정 분석"""
        message = self.input_field.get().strip()
        if message:
            # 입력 필드 초기화
            self.input_field.delete(0, tk.END)
            self.add_message(message, is_user=True)  # 사용자 메시지 추가
            
            response = self.analyze_emotion(message)  # 감정 분석 결과
            self.add_message(response, is_user=False)  # AI 응답 추가

    def add_message(self, message, is_user):
        """채팅창에 메시지 추가"""
        self.chat_area.config(state=tk.NORMAL)  # 채팅창 편집 가능 설정
        current_time = datetime.now().strftime("%H:%M")  # 현재 시간
        
        if is_user:
            # 사용자 메시지 포맷
            self.chat_area.insert(tk.END, f"사용자 ({current_time}): {message}\n")
        else:
            # AI 메시지 포맷
            self.chat_area.insert(tk.END, f"AI ({current_time}): {message}\n")
        
        self.chat_area.see(tk.END)  # 채팅창 스크롤 최하단으로 이동
        self.chat_area.config(state=tk.DISABLED)  # 채팅창 편집 불가 설정

    def run(self):
        """프로그램 실행"""
        welcome_msg = (
            "안녕하세요! 저는 감정 분석 AI입니다.\n"
            "메시지를 입력하시면 감정을 분석해 드릴게요 😊"
        )
        self.add_message(welcome_msg, is_user=False)  # 환영 메시지
        self.window.mainloop()  # GUI 루프 실행

# 메인 실행
if __name__ == "__main__":
    try:
        app = DeepEmotionAnalyzer()
        app.run()
    except Exception as e:
        print(f"프로그램 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
