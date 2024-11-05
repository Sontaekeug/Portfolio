import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import tkinter as tk
from tkinter import ttk, scrolledtext
from datetime import datetime
from konlpy.tag import Hannanum
import json
import threading
import queue
import os
import pickle

class EmotionAnalysisModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim=5):
        super().__init__()
        # 임베딩 레이어
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding_dropout = nn.Dropout(0.2)  # 드롭아웃 감소
        
        # 단순화된 LSTM
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=1,  # 레이어 수 감소
            bidirectional=True,
            dropout=0.1,   # 드롭아웃 감소
            batch_first=True
        )
        
        # 출력 레이어
        self.dropout = nn.Dropout(0.1)  # 드롭아웃 감소
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        
    def forward(self, text):
        # 임베딩
        embedded = self.embedding(text)
        embedded = self.embedding_dropout(embedded)
        
        # LSTM
        lstm_out, _ = self.lstm(embedded)
        
        # 마지막 시점의 은닉 상태 사용
        hidden = lstm_out[:, -1, :]
        
        # 출력층
        output = self.dropout(hidden)
        output = self.fc(output)
        
        return output
    
    def init_weights(self):
        """가중치 초기화"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'conv' in name or 'fc' in name:
                    nn.init.kaiming_uniform_(param, nonlinearity='relu')
                elif 'lstm' in name:
                    nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)

def load_initial_training_data():
    """초기 학습 데이터 - 각 감정당 50개"""
    return [
        # 완전 긍정 (50개) - Label: 0
        ("인생 최고의 순간이야! 정말 너무 행복해!", 0),
        ("드디어 꿈에 그리던 취업에 성공했다! 너무 기쁘다!", 0),
        ("결혼식이 너무 완벽했어, 평생 잊지 못할 거야!", 0),
        ("시험 만점 받았어! 노력이 결실을 맺었어!", 0),
        ("드디어 내 집 마련의 꿈을 이루었어!", 0),
        ("아이가 첫 걸음마를 시작했어. 감동의 눈물이 나네.", 0),
        ("우리 팀이 대회에서 우승했어! 최고의 순간이야!", 0),
        ("10년 만에 재회한 친구와 다시 만나서 너무 감격스러워!", 0),
        ("꿈에 그리던 해외여행 티켓을 예매했어! 설레서 잠이 안 와!", 0),
        ("마라톤 완주했어! 인생 최고의 도전이었어!", 0),
        ("프로포즈 성공했어! 그녀가 눈물을 흘리며 고개를 끄덕였어!", 0),
        ("드디어 박사 학위를 받았어! 모든 게 꿈만 같아!", 0),
        ("우리 아이가 첫 단어를 말했어! 이런 기쁨이!", 0),
        ("창업한 회사가 대박났어! 직원들과 함께 축하했어!", 0),
        ("봉사활동에서 만난 아이들이 행복해하는 모습에 가슴이 벅차!", 0),
        ("오늘 깜짝 생일파티를 받았어. 친구들 덕분에 최고의 생일이야!", 0),
        ("드디어 내 책이 출간되었어! 작가의 꿈을 이루었어!", 0),
        ("올림픽 금메달이야! 국민들과 함께 기뻐할 수 있어 영광이야!", 0),
        ("첫 월급으로 부모님께 선물을 드렸어. 너무 기뻐하셔서 나도 행복해!", 0),
        ("20년 만에 가족 모두가 한자리에 모였어. 이런 날이 올 줄이야!", 0),
        ("꿈에 그리던 대학교에 합격했어! 부모님이 자랑스러워하셔!", 0),
        ("사랑하는 사람과 결혼식 날짜를 잡았어! 가슴이 터질 것 같아!", 0),
        ("인생 첫 수상소감을 말했어. 모두가 축하해줘서 감격스러워!", 0),
        ("새로운 집으로 이사했어. 모든 것이 완벽해!", 0),
        ("오늘 아들이 첫 등교를 했어. 자랑스러워 눈물이 나!", 0),
        ("회사에서 특별 승진했어! 노력을 인정받은 것 같아 기뻐!", 0),
        ("오랫동안 준비한 공연이 대성공이야! 관객들의 기립박수를 받았어!", 0),
        ("드디어 자격증 시험에 합격했어! 열심히 준비한 보람이 있어!", 0),
        ("새로운 반려동물을 입양했어. 이런 행복이 있다니!", 0),
        ("팀원들과 함께 프로젝트 대상을 받았어! 최고의 순간이야!", 0),
        ("꿈에 그리던 집을 계약했어! 모든 게 완벽해!", 0),
        ("아이가 첫 그림을 그려줬어. 세상에서 가장 예쁜 그림이야!", 0),
        ("기부금으로 아이들의 학교를 지었어. 이런 보람찬 일이!", 0),
        ("영화제에서 신인상을 받았어! 꿈만 같은 순간이야!", 0),
        ("첫 사업이 대박났어! 직원들과 함께 축하했어!", 0),
        ("10년 동안 준비한 논문이 통과됐어! 이런 기쁨이!", 0),
        ("가족들과 함께하는 행복한 휴가! 평생 잊지 못할 거야!", 0),
        ("꿈에 그리던 스포츠카를 구입했어! 인생의 터닝포인트야!", 0),
        ("우리 팀이 역전승으로 우승했어! 최고의 경기였어!", 0),
        ("첫 전시회가 성황리에 마무리됐어! 모든 작품이 완판됐어!", 0),
        ("새로운 사업 아이템으로 대박났어! 직원들 연봉도 올려줄 수 있어!", 0),
        ("아이가 전국대회에서 1등했어! 자랑스러워 눈물이 나!", 0),
        ("드디어 로또 당첨됐어! 이제 꿈에 그리던 삶을 살 수 있어!", 0),
        ("새로운 앱 출시했는데 대박났어! 다운로드 수가 폭발적이야!", 0),
        ("20년 만에 잃어버린 가족을 찾았어! 기적이 일어났어!", 0),
        ("첫 해외공연이 대성공이야! 스탠딩 오베이션을 받았어!", 0),
        ("드디어 숙원사업이 성공했어! 직원들 모두 보너스 줄 수 있어!", 0),
        ("새로운 발명품이 특허 등록됐어! 전 세계가 주목하고 있어!", 0),
        ("우리 아이가 영재로 선발됐어! 이런 자랑스러운 순간이!", 0),
        ("평생의 목표였던 등반에 성공했어! 히말라야 정상에서 울었어!", 0),

        # 긍정 (50개) - Label: 1
        ("오늘 하루가 참 좋았어. 기분이 좋네.", 1),
        ("맛있는 점심 먹고 산책했더니 상쾌하다.", 1),
        ("새로 산 옷이 잘 어울려서 기분이 좋아.", 1),
        ("친구랑 수다 떨면서 커피 마시니 좋네.", 1),
        ("운동 후에 샤워하니 상쾌하고 기분 좋아.", 1),
        ("책상 정리했더니 마음이 깔끔해졌어.", 1),
        ("오늘 머리 잘 잘랐네. 기분이 새로워.", 1),
        ("동료가 커피를 사줘서 기분이 좋네.", 1),
        ("퇴근 후에 영화 보니 즐겁다.", 1),
        ("주말에 잘 쉬어서 컨디션이 좋아.", 1),
        ("집에서 만든 요리가 맛있게 됐어.", 1),
        ("좋아하는 음악 들으니 기분이 좋네.", 1),
        ("가벼운 운동하고 와서 상쾌해.", 1),
        ("친구한테 선물 받아서 기쁘네.", 1),
        ("날씨가 좋아서 기분도 좋아.", 1),
        ("새로 배운 게 있어서 뿌듯해.", 1),
        ("오랜만에 가족과 식사해서 좋았어.", 1),
        ("집 청소하고 나니 마음이 편안해.", 1),
        ("반려동물이 재롱부리니 웃음이 나.", 1),
        ("작은 칭찬을 받아서 기분이 좋네.", 1),
        ("취미생활이 잘 맞아서 즐거워.", 1),
        ("계획했던 일을 잘 마무리했어.", 1),
        ("동네 산책하니 마음이 평화로워.", 1),
        ("좋은 향기가 나서 기분이 좋아.", 1),
        ("새로운 것을 배워서 재미있어.", 1),
        ("오늘 하루 잘 보낸 것 같아 뿌듯해.", 1),
        ("좋은 사람들과 시간을 보내서 즐거워.", 1),
        ("작은 성취감이 있어서 기분이 좋아.", 1),
        ("맛있는 디저트를 먹어서 행복해.", 1),
        ("좋아하는 노래가 나와서 신나.", 1),
        ("햇살이 따뜻해서 기분이 좋네.", 1),
        ("새로운 친구를 사귀어서 즐거워.", 1),
        ("좋은 책을 읽어서 마음이 풍요로워.", 1),
        ("작은 선물을 받아서 기쁘네.", 1),
        ("일이 순조롭게 진행되어서 다행이야.", 1),
        ("좋은 아이디어가 떠올라서 신나.", 1),
        ("오랜만에 푹 자서 개운해.", 1),
        ("주변 사람들이 친절해서 좋네.", 1),
        ("계획대로 일이 진행되어서 만족스러워.", 1),
        ("작은 취미를 가져서 즐거워.", 1),
        ("새로운 장소를 발견해서 신기해.", 1),
        ("좋은 향수를 발견해서 기분이 좋아.", 1),
        ("맛있는 음식을 나눠 먹어서 즐거워.", 1),
        ("작은 운동을 시작해서 뿌듯해.", 1),
        ("새로운 음악을 발견해서 기분이 좋아.", 1),
        ("친구의 안부 연락을 받아서 반가워.", 1),
        ("좋은 글을 읽어서 감동받았어.", 1),
        ("작은 변화를 시도해서 신선해.", 1),
        ("오늘 하루 잘 마무리해서 만족스러워.", 1),
        ("좋은 생각이 떠올라서 기쁘네.", 1),

        # 중립 (50개) - Label: 2
        ("오늘은 평소와 같은 하루네.", 2),
        ("특별한 일은 없었어.", 2),
        ("그냥 보통의 하루였어.", 2),
        ("할 일을 하고 있어.", 2),
        ("일상적인 업무를 처리 중이야.", 2),
        ("평범한 아침이야.", 2),
        ("점심 시간이 다가오네.", 2),
        ("버스를 기다리고 있어.", 2),
        ("이메일을 확인하고 있어.", 2),
        ("보고서를 작성하고 있어.", 2),
        ("회의 준비를 하고 있어.", 2),
        ("자료를 정리하고 있어.", 2),
        ("일정을 확인하고 있어.", 2),
        ("전화를 기다리고 있어.", 2),
        ("문서를 검토하고 있어.", 2),
        ("커피를 마시고 있어.", 2),
        ("출근 준비 중이야.", 2),
        ("퇴근 시간이 다가와.", 2),
        ("날씨가 평범하네.", 2),
        ("일상적인 대화를 나누고 있어.", 2),
        ("평소처럼 지내고 있어.", 2),
        ("특별할 것 없는 하루네.", 2),
        ("일을 진행하고 있어.", 2),
        ("평범한 하루를 보내고 있어.", 2),
        ("일상적인 업무야.", 2),
        ("점심 식사를 했어.", 2),
        ("저녁 식사 준비 중이야.", 2),
        ("내일 일정을 보고 있어.", 2),
        ("평소와 다름없이 지내.", 2),
        ("일과를 마무리하고 있어.", 2),
        ("주간 보고를 작성 중이야.", 2),
        ("회의록을 정리하고 있어.", 2),
        ("일정 조율 중이야.", 2),
        ("평범한 주말이야.", 2),
        ("일상적인 대화를 나누고 있어.", 2),
        ("특별한 계획은 없어.", 2),
        ("평소처럼 출근했어.", 2),
        ("일과가 시작됐어.", 2),
        ("업무 관련 연락 중이야.", 2),
        ("자료 검토가 필요해.", 2),
        ("미팅 일정을 잡고 있어.", 2),
        ("일반적인 상황이야.", 2),
        ("오늘도 평소처럼 지내.", 2),
        ("별다른 일은 없어.", 2),
        ("평범한 하루를 보내고 있어.", 2),
        ("일상적인 대화 중이야.", 2),
        ("보통의 하루를 보내.", 2),
        ("특별한 변화는 없어.", 2),
        ("일상이 계속되네.", 2),

        # 부정 (50개) - Label: 3
        ("오늘은 기분이 별로야.", 3),
        ("일이 잘 안 풀려서 속상해.", 3),
        ("피곤하고 짜증나는 하루야.", 3),
        ("실수를 해서 걱정이야.", 3),
        ("마음이 무겁네.", 3),
        ("결과가 좋지 않아서 실망스러워.", 3),
        ("계획대로 안 돼서 불안해.", 3),
        ("스트레스가 쌓이는 것 같아.", 3),
        ("사람들이 내 말을 이해 못 하는 것 같아.", 3),
        ("뭔가 잘못될 것 같은 느낌이야.", 3),
        ("기대했던 것보다 안 좋네.", 3),
        ("몸이 좀 안 좋은 것 같아.", 3),
        ("일이 꼬이기 시작했어.", 3),
        ("뭔가 불안한 느낌이 들어.", 3),
        ("오늘따라 의욕이 없네.", 3),
        ("집중이 잘 안 되는 날이야.", 3),
        ("약속을 못 지켜서 미안해.", 3),
        ("결과가 마음에 들지 않아.", 3),
        ("예상보다 잘 안됐어.", 3),
        ("걱정되는 일이 생겼어.", 3),
        ("일이 복잡해져서 힘들어.", 3),
        ("마음대로 안 되니까 답답해.", 3),
        ("문제가 해결되지 않아 고민이야.", 3),
        ("기분이 좋지 않은 하루네.", 3),
        ("생각보다 어려운 일이야.", 3),
        ("조금 우울한 것 같아.", 3),
        ("상황이 안 좋아지고 있어.", 3),
        ("무언가 잘못된 것 같아.", 3),
        ("기분이 가라앉네.", 3),
        ("노력한 만큼 안 나와서 속상해.", 3),
        ("기대했던 것과 달라서 실망이야.", 3),
        ("문제가 생겨서 걱정이야.", 3),
        ("일이 꼬이기 시작했어.", 3),
        ("마음이 불편해.", 3),
        ("잘못된 선택을 한 것 같아.", 3),
        ("결과가 좋지 않아 걱정이야.", 3),
        ("스트레스 받는 상황이야.", 3),
        ("기분이 좋지 않아.", 3),
        ("계획이 틀어져서 난감해.", 3),
        ("예상치 못한 문제가 생겼어.", 3),
        ("상황이 복잡해져서 걱정이야.", 3),
        ("뭔가 잘못될 것 같아.", 3),
        ("일이 안 풀려서 답답해.", 3),
        ("걱정되는 일이 있어.", 3),
        ("마음이 좋지 않아.", 3),
        ("기분이 별로야.", 3),
        ("뭔가 불안해.", 3),
        ("생각대로 안 되네.", 3),
        ("힘든 하루였어.", 3),
        ("걱정이 많아졌어.", 3),

        # 완전 부정 (50개) - Label: 4
        ("인생이 너무 힘들어서 포기하고 싶어.", 4),
        ("모든 게 무너진 것 같아 절망스러워.", 4),
        ("더 이상 희망이 없는 것 같아.", 4),
        ("이렇게 살아서 뭐하나 싶어.", 4),
        ("너무 큰 실패를 해서 회복할 수 없을 것 같아.", 4),
        ("모든 것이 내 잘못인 것 같아 괴로워.", 4),
        ("살아갈 의미를 못 찾겠어.", 4),
        ("이제 더 이상 버틸 수가 없어.", 4),
        ("모든 게 끝난 것 같아 죽고 싶어.", 4),
        ("아무도 날 이해하지 못해 정말 외로워.", 4),
        ("세상에 혼자 버려진 것 같아.", 4),
        ("이 고통에서 벗어날 수 없을 것 같아.", 4),
        ("모든 게 무의미하게 느껴져.", 4),
        ("더 이상 살아갈 이유를 모르겠어.", 4),
        ("끝없는 절망감에 빠져있어.", 4),
        ("삶이 너무 고통스러워.", 4),
        ("모든 희망이 사라졌어.", 4),
        ("이 상황에서 벗어날 수 없을 것 같아.", 4),
        ("누구도 날 도와줄 수 없을 거야.", 4),
        ("완전히 실패자가 된 것 같아.", 4),
        ("다시는 일어설 수 없을 것 같아.", 4),
        ("모든 게 끝장난 것 같아.", 4),
        ("살아있는 게 너무 고통스러워.", 4),
        ("이제 더 이상 견딜 수 없어.", 4),
        ("극심한 우울감에 시달려.", 4),
        ("인생이 완전히 망가져버렸어.", 4),
        ("모든 게 끝난 것 같아 괴로워.", 4),
        ("절망 속에서 헤어나올 수 없어.", 4),
        ("삶이 너무 버거워.", 4),
        ("모든 걸 포기하고 싶어.", 4),
        ("더 이상 살아갈 의미가 없어.", 4),
        ("세상이 너무 잔인해.", 4),
        ("모든 게 너무 힘들어.", 4),
        ("살아있는 것 자체가 고통이야.", 4),
        ("이 고통이 영원할 것 같아.", 4),
        ("완전히 무너져버렸어.", 4),
        ("삶이 너무 공허해.", 4),
        ("모든 게 끝난 것 같아.", 4),
        ("더 이상 희망이 보이지 않아.", 4),
        ("극도의 절망감을 느껴.", 4),
        ("살아갈 이유를 잃어버렸어.", 4),
        ("모든 게 의미 없어 보여.", 4),
        ("이 고통에서 벗어나고 싶어.", 4),
        ("삶이 너무 괴로워.", 4),
        ("더 이상 버틸 수가 없어.", 4),
        ("모든 것이 절망적이야.", 4),
        ("살아가는 게 너무 힘들어.", 4),
        ("완전히 포기하고 싶어.", 4),
        ("이제 더 이상 못하겠어.", 4),
        ("모든 게 끝이야.", 4)
    ]
    
class DeepEmotionAnalyzer:
    def __init__(self):
        # 데이터 저장 경로
        self.data_dir = "emotion_analyzer_data"
        self.vocab_path = os.path.join(self.data_dir, "vocab.json")
        self.model_path = os.path.join(self.data_dir, "model.pth")
        self.training_data_path = os.path.join(self.data_dir, "training_data.pkl")
        
        # 디렉토리 생성
        os.makedirs(self.data_dir, exist_ok=True)
        
        # 하이퍼파라미터
        self.vocab_size = 5000
        self.embedding_dim = 100
        self.hidden_dim = 64
        self.output_dim = 5  # 5개 감정 카테고리
        
        # 형태소 분석기 초기화 (변경된 부분)
        self.tokenizer = Hannanum()
        
        # 초기화
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
            else:
                self.word_to_idx = {'<PAD>': 0, '<UNK>': 1}
                self.idx_to_word = {0: '<PAD>', 1: '<UNK>'}
        except Exception as e:
            print(f"어휘 사전 로드 중 오류: {e}")
            self.word_to_idx = {'<PAD>': 0, '<UNK>': 1}
            self.idx_to_word = {0: '<PAD>', 1: '<UNK>'}

    def load_training_data(self):
        """학습 데이터 로드"""
        try:
            if os.path.exists(self.training_data_path):
                with open(self.training_data_path, 'rb') as f:
                    self.train_data = pickle.load(f)
            else:
                self.train_data = []
                # 초기 학습 데이터 추가
                self.train_data.extend(load_initial_training_data())
                self.save_training_data()
        except Exception as e:
            print(f"학습 데이터 로드 중 오류: {e}")
            self.train_data = []

    def save_training_data(self):
        """학습 데이터 저장"""
        try:
            with open(self.training_data_path, 'wb') as f:
                pickle.dump(self.train_data, f)
        except Exception as e:
            print(f"학습 데이터 저장 중 오류: {e}")

    def save_vocab(self):
        """어휘 사전 저장"""
        try:
            with open(self.vocab_path, 'w', encoding='utf-8') as f:
                json.dump(self.word_to_idx, f, ensure_ascii=False)
        except Exception as e:
            print(f"어휘 사전 저장 중 오류: {e}")

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
        except Exception as e:
            print(f"모델 저장 중 오류: {e}")

    def initialize_model(self):
        """모델 초기화 또는 로드"""
        try:
            if os.path.exists(self.model_path):
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
                self.model = EmotionAnalysisModel(
                    self.vocab_size,
                    self.embedding_dim,
                    self.hidden_dim,
                    self.output_dim
                )
                self.model.init_weights()  # 가중치 초기화 추가
                print("새로운 모델을 초기화했습니다.")
        except Exception as e:
            print(f"모델 초기화 중 오류: {e}")
            self.model = EmotionAnalysisModel(
                self.vocab_size,
                self.embedding_dim,
                self.hidden_dim,
                self.output_dim
            )
            self.model.init_weights()  # 가중치 초기화 추가
        
    def setup_gui(self):
        """GUI 설정"""
        self.window = tk.Tk()
        self.window.title("감정 분석 채팅")
        self.window.geometry("500x700")
        self.window.configure(bg='#BACEE0')  # 배경색 설정
        
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
        
        # 채팅 영역 생성 및 스타일 설정
        self.chat_area = tk.Text(
            self.chat_frame,
            wrap=tk.WORD,
            width=50,
            height=20,
            font=("맑은 고딕", 10),
            background="#BACEE0",  # 배경색
            relief=tk.FLAT,
            padx=15,
            pady=15,
            highlightthickness=0,  # 포커스 테두리 제거
            insertwidth=0,  # 커서 숨기기
            selectbackground="#A0A0A0",  # 선택 영역 색상
            spacing1=2,  # 위 여백
            spacing2=2,  # 아래 여백
            spacing3=0   # 줄 간격
        )
        self.chat_area.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
        
        # 스크롤바
        scrollbar = ttk.Scrollbar(self.chat_frame, orient=tk.VERTICAL, command=self.chat_area.yview)
        scrollbar.pack(fill=tk.Y, side=tk.RIGHT)
        self.chat_area.configure(yscrollcommand=scrollbar.set)
        
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
            command=self.send_message
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
        emotions = [
            ("완전 긍정", "very_positive"),
            ("긍정", "positive"),
            ("중립", "neutral"),
            ("부정", "negative"),
            ("완전 부정", "very_negative")
        ]
        
        # 감정 라디오 버튼을 두 줄로 배치
        button_frame1 = ttk.Frame(emotion_frame)
        button_frame1.pack(fill=tk.X)
        button_frame2 = ttk.Frame(emotion_frame)
        button_frame2.pack(fill=tk.X)
        
        for i, (text, value) in enumerate(emotions):
            if i < 3:  # 첫 번째 줄
                ttk.Radiobutton(
                    button_frame1,
                    text=text,
                    value=value,
                    variable=self.emotion_var
                ).pack(side=tk.LEFT, padx=5)
            else:  # 두 번째 줄
                ttk.Radiobutton(
                    button_frame2,
                    text=text,
                    value=value,
                    variable=self.emotion_var
                ).pack(side=tk.LEFT, padx=5)
        
        # 버튼들
        button_frame = ttk.Frame(training_frame, style="Controls.TFrame")
        button_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(
            button_frame,
            text="예제 추가",
            command=self.add_training_example
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            button_frame,
            text="모델 학습",
            command=self.train_model
        ).pack(side=tk.LEFT, padx=5)
        
        # 상태 표시
        self.status_var = tk.StringVar(value="준비됨")
        self.status_label = ttk.Label(
            training_frame,
            textvariable=self.status_var,
            style="Controls.TLabel"
        )
        self.status_label.pack(fill=tk.X, pady=5)
    
    def add_training_example(self):
        """학습 예제 추가"""
        text = self.input_field.get().strip()
        emotion = self.emotion_var.get()
        
        if text:
            # 감정 레이블 변환
            emotion_to_label = {
                "very_positive": 0,   # 완전 긍정
                "positive": 1,        # 긍정
                "neutral": 2,         # 중립
                "negative": 3,        # 부정
                "very_negative": 4    # 완전 부정
            }
            label = emotion_to_label[emotion]
            
            # 감정 한글 변환
            emotion_korean = {
                "very_positive": "완전 긍정",
                "positive": "긍정",
                "neutral": "중립",
                "negative": "부정",
                "very_negative": "완전 부정"
            }
            
            # 데이터 추가
            self.train_data.append((text, label))
            self.save_training_data()
            self.update_status()
            self.input_field.delete(0, tk.END)
            
            # 알림 메시지 추가
            self.add_message(f"학습 예제가 추가되었습니다. (감정: {emotion_korean[emotion]})", is_user=False)
    
    def add_message(self, message, is_user=True):
        """채팅창에 메시지 추가"""
        self.chat_area.config(state=tk.NORMAL)
        
        # 새 메시지 추가 전 줄바꿈
        if self.chat_area.get("1.0", tk.END).strip():
            self.chat_area.insert(tk.END, "\n\n")  # 메시지 간 간격

        current_time = datetime.now().strftime("%H:%M")

        if is_user:
            # 사용자 메시지 (오른쪽 정렬)
            # 여백 추가
            self.chat_area.insert(tk.END, " " * 50)
            
            # 메시지 컨테이너 시작
            msg_start = self.chat_area.index("end-1c")
            
            # 메시지 내용
            self.chat_area.insert(tk.END, f"  {message}  ", "user_message")
            msg_end = self.chat_area.index("end-1c")
            
            # 시간 표시 (메시지 아래, 오른쪽 정렬)
            self.chat_area.insert(tk.END, "\n")
            self.chat_area.insert(tk.END, " " * 45 + current_time, "time")
            
            # 말풍선 스타일 적용
            self.chat_area.tag_add("user_bubble", msg_start, msg_end)
            self.chat_area.tag_configure(
                "user_bubble",
                background="#FFEB33",
                relief="solid",
                borderwidth=0,
                lmargin1=20,
                lmargin2=20,
                rmargin=20,
                spacing1=3,  # 위 여백 줄임
                spacing2=3,  # 아래 여백 줄임
                spacing3=2,  # 줄 간격 줄임
                justify="right"
            )
            
            # 둥근 모서리 효과를 위한 스타일
            self.chat_area.tag_configure(
                "user_message",
                font=("맑은 고딕", 10),
                wrap="word",
                background="#FFEB33",
                rmargin=20,
                lmargin1=10,
                lmargin2=10,
                borderwidth=0,
                relief="solid"
            )
            
        else:
            # AI 메시지 (왼쪽 정렬)
            # 메시지 컨테이너 시작
            msg_start = self.chat_area.index("end-1c")
            
            # 메시지 내용
            self.chat_area.insert(tk.END, f"  {message}  ", "ai_message")
            msg_end = self.chat_area.index("end-1c")
            
            # 시간 표시 (메시지 아래, 왼쪽 정렬)
            self.chat_area.insert(tk.END, "\n")
            self.chat_area.insert(tk.END, " " * 5 + current_time, "time")
            
            # 말풍선 스타일 적용
            self.chat_area.tag_add("ai_bubble", msg_start, msg_end)
            self.chat_area.tag_configure(
                "ai_bubble",
                background="#FFFFFF",
                relief="solid",
                borderwidth=0,
                lmargin1=20,
                lmargin2=20,
                rmargin=20,
                spacing1=3,  # 위 여백 줄임
                spacing2=3,  # 아래 여백 줄임
                spacing3=2,  # 줄 간격 줄임
                justify="left"
            )
            
            # 둥근 모서리 효과를 위한 스타일
            self.chat_area.tag_configure(
                "ai_message",
                font=("맑은 고딕", 10),
                wrap="word",
                background="#FFFFFF",
                rmargin=10,
                lmargin1=10,
                lmargin2=10,
                borderwidth=0,
                relief="solid"
            )

        # 시간 스타일
        self.chat_area.tag_configure(
            "time",
            foreground="gray",
            font=("맑은 고딕", 8),
            spacing1=2,
            justify="right"
        )

        # 스크롤을 최하단으로
        self.chat_area.see(tk.END)
        self.chat_area.config(state=tk.DISABLED)

    def send_message(self):
        """메시지 전송"""
        message = self.input_field.get().strip()
        if message:
            self.input_field.delete(0, tk.END)
            self.add_message(message)
            
            # 감정 분석 실행
            response = self.analyze_emotion(message)
            self.add_message(response, is_user=False)

    def analyze_emotion(self, text):
        try:
            input_tensor = self.preprocess_text(text)
            
            with torch.no_grad():
                self.model.eval()
                outputs = self.model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                
                # 확률 분포 조정 (극단적인 값 방지)
                probabilities = probabilities.sqrt()  # 제곱근 적용으로 극단값 완화
                probabilities = probabilities / probabilities.sum()  # 정규화
                
                probs = probabilities[0].tolist()
                emotions = {
                    "완전 긍정": (probs[0], "🥰"),
                    "긍정": (probs[1], "😊"),
                    "중립": (probs[2], "😐"),
                    "부정": (probs[3], "😢"),
                    "완전 부정": (probs[4], "😭")
                }
                
                emotion, (prob, emoji) = max(emotions.items(), key=lambda x: x[1][0])
                
                debug_info = " | ".join([f"{e}: {p:.1%}" for e, (p, _) in emotions.items()])
                
                return f"{emotion}적인 감정이 느껴져요 {emoji}\n[세부 확률: {debug_info}]"
            
        except Exception as e:
            return f"감정 분석 중 오류가 발생했습니다: {str(e)}"

    def preprocess_text(self, text):
        """텍스트 전처리"""
        # 형태소 분석 (변경된 부분)
        morphs = self.tokenizer.morphs(text.lower())
        
        # 단어 인덱스 변환
        indexes = []
        for morph in morphs:
            if morph not in self.word_to_idx and len(self.word_to_idx) < self.vocab_size:
                self.word_to_idx[morph] = len(self.word_to_idx)
                self.idx_to_word[self.word_to_idx[morph]] = morph
            
            idx = self.word_to_idx.get(morph, self.word_to_idx['<UNK>'])
            indexes.append(idx)
        
        # 패딩
        if len(indexes) < 50:
            indexes = indexes + [0] * (50 - len(indexes))
        else:
            indexes = indexes[:50]
        
        return torch.tensor(indexes).unsqueeze(0)
    
    def train_model(self):
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
                input_tensor = self.preprocess_text(text)
                X.append(input_tensor)
                y.append(label)
            
            X = torch.cat(X)
            y = torch.tensor(y)
            
            # 데이터셋 생성
            dataset = torch.utils.data.TensorDataset(X, y)
            batch_size = 32
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True
            )
            
            # 학습 파라미터 설정
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=0.001,
                weight_decay=0.0001  # L2 정규화 감소
            )
            
            # 학습
            best_loss = float('inf')
            patience = 0
            max_patience = 10
            
            for epoch in range(100):  # 에포크 수 감소
                self.model.train()
                total_loss = 0
                
                for batch_X, batch_y in dataloader:
                    optimizer.zero_grad()
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    
                    # 그래디언트 클리핑
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    total_loss += loss.item()
                
                # 검증
                self.model.eval()
                with torch.no_grad():
                    val_outputs = self.model(X)
                    val_loss = criterion(val_outputs, y)
                    predictions = torch.softmax(val_outputs, dim=1)  # softmax 추가
                    
                    # 예측 확률 분포 확인
                    avg_probs = predictions.mean(dim=0)
                    print(f"Average probabilities: {avg_probs}")
                
                # Early stopping
                if val_loss < best_loss:
                    best_loss = val_loss
                    self.save_model()
                    patience = 0
                else:
                    patience += 1
                    if patience >= max_patience:
                        break
                
                # 상태 업데이트
                status_msg = f"Epoch {epoch+1}/100 | Loss: {total_loss/len(dataloader):.4f}"
                self.status_var.set(status_msg)
                self.window.update()
            
            self.status_var.set("학습 완료!")
            
        except Exception as e:
            self.status_var.set(f"학습 중 오류 발생: {str(e)}")
            print(f"상세 오류: {e}")

    def update_status(self):
        """상태 정보 업데이트"""
        status = f"학습 데이터: {len(self.train_data)}개 | 어휘 크기: {len(self.word_to_idx)}개"
        self.status_var.set(status)

    def run(self):
        """프로그램 실행"""
        welcome_msg = (
            "안녕하세요! 저는 딥러닝 기반 감정 분석 AI입니다.\n"
            f"현재 {len(self.train_data)}개의 학습 데이터가 있습니다.\n"
            "메시지를 입력하시면 감정을 분석해드릴게요! 😊"
        )
        self.add_message(welcome_msg, is_user=False)
        
        # 메인 루프 실행
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