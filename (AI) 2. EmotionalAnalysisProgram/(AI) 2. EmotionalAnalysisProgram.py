import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import tkinter as tk
from tkinter import ttk, scrolledtext
from datetime import datetime
from konlpy.tag import Okt
import json
import threading
import queue
import os
import pickle
import re
from sklearn.metrics import precision_recall_fscore_support

class EmotionAnalysisModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        # 임베딩 레이어 설정
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding_dropout = nn.Dropout(0.2)  # 드롭아웃 값 설정
        
        # CNN 레이어 설정
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(embedding_dim, hidden_dim, kernel_size=k, padding=k//2)
            for k in [3, 4, 5]
        ])
        
        # LSTM 레이어 설정
        self.lstm = nn.LSTM(
            hidden_dim * 3,  # 3개의 컨볼루션 결과를 연결
            hidden_dim,
            num_layers=2,
            bidirectional=True,
            dropout=0.3,
            batch_first=True
        )
        
        # 어텐션 메커니즘 설정
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(),  # ReLU 대신 LeakyReLU 사용 0 이하 값 기울기 값 적용(* 0.1)
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1)
        )
        
        # 출력 레이어
        self.dropout = nn.Dropout(0.3)  # 드롭아웃 비율 조정
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.leaky_relu = nn.LeakyReLU()
        self.batch_norm = nn.BatchNorm1d(hidden_dim)  # 배치 정규화 추가
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, text):
        # 임베딩
        embedded = self.embedding(text)
        embedded = self.embedding_dropout(embedded)
        
        # CNN 처리
        conv_input = embedded.transpose(1, 2)
        conv_outputs = []
        for conv in self.conv_layers:
            conv_out = self.leaky_relu(conv(conv_input))
            pooled = F.max_pool1d(conv_out, conv_out.shape[2]).squeeze(2)
            conv_outputs.append(pooled)
        
        conv_output = torch.cat(conv_outputs, dim=1)
        conv_output = conv_output.unsqueeze(1).repeat(1, embedded.size(1), 1)
        
        # LSTM 처리
        lstm_out, _ = self.lstm(conv_output)
        
        # 어텐션 적용
        attention_weights = self.attention(lstm_out)
        attention_output = torch.bmm(lstm_out.transpose(1, 2), attention_weights).squeeze(-1)
        
        # 최종 출력
        output = self.dropout(attention_output)
        output = self.fc1(output)
        output = self.leaky_relu(output)
        output = self.batch_norm(output)
        output = self.dropout(output)
        output = self.fc2(output)
        
        return output

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

class DeepEmotionAnalyzer:
    def __init__(self):
        # 데이터 저장 경로
        self.data_dir = "emotion_analyzer_data"
        self.vocab_path = os.path.join(self.data_dir, "vocab.json")
        self.model_path = os.path.join(self.data_dir, "model.pth")
        self.training_data_path = os.path.join(self.data_dir, "training_data.pkl")
        
        # 감정 키워드 사전 초기화
        self.emotion_keywords = {
            'positive': ['좋아', '행복', '기쁘', '감사', '축하', '사랑', '즐겁', '신나', '훌륭', '멋지'],
            'negative': ['슬프', '힘들', '아프', '싫어', '미안', '속상', '화나', '괴롭', '실망', '후회'],
            'neutral': ['보통', '평범', '일반', '보편', '무난', '적당', '보편', '중간', '그저', '담담']
        }
        
        os.makedirs(self.data_dir, exist_ok=True)
        
        # 하이퍼파라미터 설정정
        self.vocab_size = 8000  # 어휘 크기 설정
        self.embedding_dim = 200  # 임베딩 차원 설정
        self.hidden_dim = 128  # 은닉층 차원 설정
        self.output_dim = 3
        
        self.okt = Okt()
        
        self.load_vocab()
        self.load_training_data()
        self.initialize_model()
        self.setup_gui()

    def load_vocab(self):
        """어휘 사전 로드"""
        try:
            if os.path.exists(self.vocab_path):
                with open(self.vocab_path, 'r', encoding='utf-8') as f:
                    self.word_to_idx = json.load(f)
                    self.idx_to_word = {v: k for k, v in self.word_to_idx.items()}
                    print("어휘 사전을 로드했습니다.")
            else:
                self.word_to_idx = {'<PAD>': 0, '<UNK>': 1}
                self.idx_to_word = {0: '<PAD>', 1: '<UNK>'}
                print("새로운 어휘 사전을 생성했습니다.")
        except Exception as e:
            print(f"어휘 사전 로드 중 오류: {e}")
            self.word_to_idx = {'<PAD>': 0, '<UNK>': 1}
            self.idx_to_word = {0: '<PAD>', 1: '<UNK>'}

    def save_vocab(self):
        """어휘 사전 저장"""
        try:
            with open(self.vocab_path, 'w', encoding='utf-8') as f:
                json.dump(self.word_to_idx, f, ensure_ascii=False, indent=2)
                print("어휘 사전을 저장했습니다.")
        except Exception as e:
            print(f"어휘 사전 저장 중 오류: {e}")

    def load_training_data(self):
        """학습 데이터 로드"""
        try:
            if os.path.exists(self.training_data_path):
                with open(self.training_data_path, 'rb') as f:
                    self.train_data = pickle.load(f)
                    print(f"학습 데이터 {len(self.train_data)}개를 로드했습니다.")
            else:
                self.train_data = []
                # 초기 학습 데이터 추가
                self.train_data.extend(load_initial_training_data())
                self.save_training_data()
                print(f"초기 학습 데이터 {len(self.train_data)}개를 생성했습니다.")
        except Exception as e:
            print(f"학습 데이터 로드 중 오류: {e}")
            self.train_data = []

    def save_training_data(self):
        """학습 데이터 저장"""
        try:
            with open(self.training_data_path, 'wb') as f:
                pickle.dump(self.train_data, f)
                print("학습 데이터를 저장했습니다.")
        except Exception as e:
            print(f"학습 데이터 저장 중 오류: {e}")

    def save_model(self):
        """모델 저장"""
        try:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'vocab_size': self.vocab_size,
                'embedding_dim': self.embedding_dim,
                'hidden_dim': self.hidden_dim,
                'output_dim': self.output_dim
            }, self.model_path)
            print("모델을 저장했습니다.")
        except Exception as e:
            print(f"모델 저장 중 오류: {e}")
            
    def initialize_model(self):
        """모델 초기화 또는 로드"""
        try:
            if os.path.exists(self.model_path):
                # 기존 모델 로드
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
            
            # GPU 사용 가능한 경우 GPU로 이동 //CUDA 설치 안 됐을 경우 CPU로로
            if torch.cuda.is_available():
                self.model = self.model.cuda()
                print("GPU를 사용합니다.")
            else:
                print("CPU를 사용합니다.")
                
        except Exception as e:
            print(f"모델 초기화 중 오류 발생: {e}")
            # 기존 파일 없을 시 새 모델 생성
            self.model = EmotionAnalysisModel(
                self.vocab_size,
                self.embedding_dim,
                self.hidden_dim,
                self.output_dim
            )
            print("오류로 인해 새 모델을 생성했습니다.")

    def get_device(self):
        """현재 사용 가능한 디바이스 반환"""
        if torch.cuda.is_available():
            return torch.device('cuda')
        return torch.device('cpu')

    def move_to_device(self, tensor):
        """텐서를 적절한 디바이스로 이동"""
        device = self.get_device()
        return tensor.to(device)
    
    def improved_preprocess_text(self, text):
        """향상된 텍스트 전처리"""
        # 텍스트 정규화
        text = text.lower()
        text = re.sub(r'[^가-힣a-z\s]', ' ', text)
        
        # 형태소 분석
        morphs = self.okt.morphs(text)
        
        # 감정 키워드 가중치 반영
        weighted_morphs = []
        for morph in morphs:
            weight = 1.0
            
            # 감정 키워드에 가중치 부여
            if any(keyword in morph for keyword in self.emotion_keywords['positive']):
                weight = 1.5
            elif any(keyword in morph for keyword in self.emotion_keywords['negative']):
                weight = 1.5
            elif any(keyword in morph for keyword in self.emotion_keywords['neutral']):
                weight = 1.2
                
            # 단어 인덱스 변환
            if morph not in self.word_to_idx and len(self.word_to_idx) < self.vocab_size:
                self.word_to_idx[morph] = len(self.word_to_idx)
                self.idx_to_word[self.word_to_idx[morph]] = morph
            
            idx = self.word_to_idx.get(morph, self.word_to_idx['<UNK>'])
            weighted_morphs.extend([idx] * int(weight))
        
        # 동적 패딩 적용
        max_length = 100
        if len(weighted_morphs) < max_length:
            weighted_morphs = weighted_morphs + [0] * (max_length - len(weighted_morphs))
        else:
            weighted_morphs = weighted_morphs[:max_length]
        
        return torch.tensor(weighted_morphs).unsqueeze(0)

    def train_model(self):
        """입력된 모델 학습"""
        if len(self.train_data) < 5:
            self.status_var.set("최소 5개의 학습 예제가 필요합니다")
            return
        
        self.status_var.set("학습 중...")
        self.window.update()
        
        try:
            # 데이터 준비
            X = []
            y = []
            for text, label in self.train_data:
                input_tensor = self.improved_preprocess_text(text)
                X.append(input_tensor)
                y.append(label)
            
            X = torch.cat(X)
            y = torch.tensor(y)
            
            # 데이터셋 생성
            dataset = torch.utils.data.TensorDataset(X, y)
            batch_size = min(32, len(dataset))
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True
            )
            
            # 클래스 가중치 계산 및 적용
            class_counts = torch.bincount(y, minlength=3)
            weights = 1.0 / class_counts.float()
            weights = weights / weights.sum()
            weights = weights * torch.tensor([1.5, 1.0, 1.2])  # 긍정, 중립, 부정 클래스 가중치 조정
            
            criterion = nn.CrossEntropyLoss(weight=weights)
            
            # 최적화기 설정
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=0.001,
                weight_decay=0.001,
                betas=(0.9, 0.999)
            )
            
            # 학습률 스케줄러
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                verbose=True
            )
            
            # 학습
            best_loss = float('inf')
            patience = 0
            max_patience = 15  # Early stopping 설정
            
            for epoch in range(150):  # 에폭 수 증가
                self.model.train()
                total_loss = 0
                predictions = []
                true_labels = []
                
                for batch_X, batch_y in dataloader:
                    optimizer.zero_grad()
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    
                    # 그래디언트 클리핑
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    
                    total_loss += loss.item()
                    _, predicted = outputs.max(1)
                    predictions.extend(predicted.tolist())
                    true_labels.extend(batch_y.tolist())
                
                # 평가 메트릭 계산
                precision, recall, f1, _ = precision_recall_fscore_support(
                    true_labels,
                    predictions,
                    average=None,
                    labels=[0, 1, 2]
                )
                
                # 검증
                self.model.eval()
                with torch.no_grad():
                    val_outputs = self.model(X)
                    val_loss = criterion(val_outputs, y)
                    val_acc = (val_outputs.argmax(1) == y).float().mean()
                
                # 학습률 조정
                scheduler.step(val_loss)
                
                # 상태 업데이트
                epoch_loss = total_loss / len(dataloader)
                status_msg = (
                    f"Epoch {epoch+1}/150 | Loss: {epoch_loss:.4f} | "
                    f"Val Acc: {val_acc:.2f} | F1: {f1.mean():.2f}"
                )
                self.status_var.set(status_msg)
                self.window.update()
                
                # Early stopping 검사
                if val_loss < best_loss:
                    best_loss = val_loss
                    self.save_model()
                    patience = 0
                else:
                    patience += 1
                    if patience >= max_patience:
                        break
            
            self.status_var.set("학습 완료!")
            
        except Exception as e:
            self.status_var.set(f"학습 중 오류 발생: {str(e)}")
            print(f"상세 오류: {e}")

    def analyze_emotion(self, text):
        """감정 분석"""
        try:
            input_tensor = self.improved_preprocess_text(text)
            
            with torch.no_grad():
                self.model.eval()
                outputs = self.model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                
                # 감정 키워드 기반 보정
                keyword_scores = {
                    'positive': 0,
                    'negative': 0,
                    'neutral': 0
                }
                
                # 텍스트에서 감정 키워드 검출
                for emotion, keywords in self.emotion_keywords.items():
                    for keyword in keywords:
                        if keyword in text:
                            keyword_scores[emotion] += 0.1
                
                # 최종 확률 계산
                probs = probabilities[0].tolist()
                probs[0] += keyword_scores['positive']  # 긍정
                probs[1] += keyword_scores['neutral']   # 중립
                probs[2] += keyword_scores['negative']  # 부정
                
                # 정규화
                total = sum(probs)
                probs = [p/total for p in probs]
                
                emotions = {
                    "긍정": (probs[0], "😊"),
                    "중립": (probs[1], "😐"),
                    "부정": (probs[2], "😢")
                }
                
                # 가장 높은 확률의 감정 선택
                emotion, (prob, emoji) = max(emotions.items(), key=lambda x: x[1][0])
                
                # 상세 확률 출력
                detail_info = " | ".join([f"{e}: {p:.1%}" for e, (p, _) in emotions.items()])
                
                # 신뢰도 수준 추가
                confidence = max(probs)
                confidence_level = "높음" if confidence > 0.6 else "중간" if confidence > 0.4 else "낮음"
                
                return f"{emotion}적인 감정이 느껴져요 {emoji}\n[신뢰도 {confidence_level} | {detail_info}]"
            
        except Exception as e:
            return f"감정 분석 중 오류가 발생했습니다: {str(e)}"

    # GUI 관련 메서드
    def setup_gui(self):
        """GUI 설정"""
        self.window = tk.Tk()
        self.window.title("감정 분석 채팅")
        self.window.geometry("500x700")
        self.window.configure(bg='#BACEE0')  # 메인 윈도우 배경색
        
        style = ttk.Style()
        style.configure("Chat.TFrame", background="#BACEE0")
        style.configure("Controls.TFrame", background="#BACEE0")
        style.configure("Controls.TLabelframe", background="#BACEE0")
        style.configure("Controls.TLabel", background="#BACEE0")
        style.configure("Controls.TButton", padding=5)
        
        self.main_frame = ttk.Frame(self.window, style="Chat.TFrame")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        self.setup_chat_area()
        self.setup_input_area()
        self.setup_training_controls()
        self.update_status()

    def setup_chat_area(self):
        """채팅 영역 설정"""
        self.chat_frame = ttk.Frame(self.main_frame, style="Chat.TFrame")
        self.chat_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.chat_area = tk.Text(
            self.chat_frame,
            wrap=tk.WORD,
            width=50,
            height=20,
            font=("맑은 고딕", 10),
            background="#BACEE0",  # 채팅 영역 배경색
            relief=tk.FLAT,
            padx=10,
            pady=10
        )
        self.chat_area.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
        
        scrollbar = ttk.Scrollbar(self.chat_frame, orient=tk.VERTICAL, command=self.chat_area.yview)
        scrollbar.pack(fill=tk.Y, side=tk.RIGHT)
        self.chat_area.configure(yscrollcommand=scrollbar.set)
        
        self.chat_area.tag_configure("time", foreground="gray", font=("맑은 고딕", 8))
        self.chat_area.tag_configure("user_message", font=("맑은 고딕", 10), justify="right")
        self.chat_area.tag_configure("ai_message", font=("맑은 고딕", 10), justify="left")
        self.chat_area.tag_configure("system", foreground="blue", font=("맑은 고딕", 9))
        
        self.chat_area.config(state=tk.DISABLED)

    def setup_input_area(self):
        """입력 영역 설정"""
        input_frame = ttk.Frame(self.main_frame, style="Controls.TFrame")
        input_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.input_field = ttk.Entry(
            input_frame,
            font=("맑은 고딕", 10)
        )
        self.input_field.pack(fill=tk.X, side=tk.LEFT, expand=True, padx=(0, 5))
        
        send_button = ttk.Button(
            input_frame,
            text="전송",
            command=self.send_message,
            style="Controls.TButton"
        )
        send_button.pack(side=tk.RIGHT)
        
        self.input_field.bind("<Return>", lambda e: self.send_message())

    def setup_training_controls(self):
        """학습 제어 영역 설정"""
        training_frame = ttk.LabelFrame(
            self.main_frame,
            text="학습 제어",
            padding=10,
            style="Controls.TLabelframe"
        )
        training_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # 감정 선택
        emotion_frame = ttk.Frame(training_frame, style="Controls.TFrame")
        emotion_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(
            emotion_frame,
            text="감정:",
            style="Controls.TLabel"
        ).pack(side=tk.LEFT)
        
        self.emotion_var = tk.StringVar(value="neutral")
        emotions = [("긍정", "positive"), ("중립", "neutral"), ("부정", "negative")]
        
        for text, value in emotions:
            ttk.Radiobutton(
                emotion_frame,
                text=text,
                value=value,
                variable=self.emotion_var
            ).pack(side=tk.LEFT, padx=5)
        
        # 버튼 프레임
        button_frame = ttk.Frame(training_frame, style="Controls.TFrame")
        button_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(
            button_frame,
            text="예제 추가",
            command=self.add_training_example,
            style="Controls.TButton"
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            button_frame,
            text="모델 학습",
            command=self.train_model,
            style="Controls.TButton"
        ).pack(side=tk.LEFT, padx=5)

        # 모델 저장/로드 버튼 추가
        ttk.Button(
            button_frame,
            text="모델 저장",
            command=self.save_model,
            style="Controls.TButton"
        ).pack(side=tk.LEFT, padx=5)
        
        # 상태 표시
        self.status_var = tk.StringVar(value="준비됨")
        self.status_label = ttk.Label(
            training_frame,
            textvariable=self.status_var,
            style="Controls.TLabel"
        )
        self.status_label.pack(fill=tk.X, pady=5)

    def add_message(self, message, is_user=True):
        """채팅창에 메시지 추가"""
        self.chat_area.config(state=tk.NORMAL)
        
        current_time = datetime.now().strftime("%H:%M")
        
        if self.chat_area.get("1.0", tk.END).strip():
            self.chat_area.insert(tk.END, "\n\n")

        if is_user:
            self.chat_area.insert(tk.END, " " * 45)
            self.chat_area.insert(tk.END, current_time + "\n", "time")
            self.chat_area.insert(tk.END, " " * 10)
            msg_start = self.chat_area.index("end-1c")
            self.chat_area.insert(tk.END, " " + message + " ", "user_message")
            msg_end = self.chat_area.index("end-1c")
            
            self.chat_area.tag_add("user_bubble", msg_start, msg_end)
            self.chat_area.tag_configure(
                "user_bubble",
                background="#FFEB33",  # 사용자 메시지 배경색 (노란색)
                borderwidth=1,
                relief="solid",
                lmargin1=20,
                lmargin2=20,
                rmargin=20,
                justify="right"
            )
        else:
            self.chat_area.insert(tk.END, current_time + "\n", "time")
            msg_start = self.chat_area.index("end-1c")
            self.chat_area.insert(tk.END, " " + message + " ", "ai_message")
            msg_end = self.chat_area.index("end-1c")
            
            self.chat_area.tag_add("ai_bubble", msg_start, msg_end)
            self.chat_area.tag_configure(
                "ai_bubble",
                background="#BACEE0",  # AI 메시지 배경색
                borderwidth=1,
                relief="solid",
                lmargin1=20,
                lmargin2=20,
                rmargin=20,
                justify="left"
            )

        self.chat_area.see(tk.END)
        self.chat_area.config(state=tk.DISABLED)

    def send_message(self):
        """메시지 전송 처리"""
        message = self.input_field.get().strip()
        if message:
            self.input_field.delete(0, tk.END)
            self.add_message(message)
            
            response = self.analyze_emotion(message)
            self.add_message(response, is_user=False)

    def add_training_example(self):
        """학습 예제 추가"""
        text = self.input_field.get().strip()
        emotion = self.emotion_var.get()
        
        if text:
            emotion_to_label = {"positive": 0, "neutral": 1, "negative": 2}
            label = emotion_to_label[emotion]
            
            self.train_data.append((text, label))
            self.save_training_data()
            self.update_status()
            self.input_field.delete(0, tk.END)
            
            self.add_message(f"학습 예제가 추가되었습니다. (감정: {emotion})", is_user=False)

    def update_status(self):
        """상태 정보 업데이트"""
        status = f"학습 데이터: {len(self.train_data)}개 | 어휘 크기: {len(self.word_to_idx)}개"
        self.status_var.set(status)

    def run(self):
        """프로그램 실행"""
        welcome_msg = (
            "안녕하세요! 저는 감정 분석 AI입니다.\n"
            f"현재 {len(self.train_data)}개의 데이터를 학습했습니다.\n"
            "메시지를 입력하시면 감정을 분석해드릴게요! 😊\n"
        )
        self.add_message(welcome_msg, is_user=False)
        
        self.window.mainloop()

# 메인 실행 코드
if __name__ == "__main__":
    try:
        app = DeepEmotionAnalyzer()
        app.run()
    except Exception as e:
        print(f"프로그램 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
