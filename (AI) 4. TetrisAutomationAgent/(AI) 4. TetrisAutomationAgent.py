import gym  # 강화학습 환경 생성
import pygame  # GUI 및 그래픽 처리
import numpy as np  # 수학 연산 및 배열 처리
import random  # 난수 생성
import os  # 파일 작업
import gc  # 가비지 컬렉션
import time  # 시간 관리
import glob  # 파일 패턴 매칭
import pickle  # 데이터 직렬화/역직렬화
import json  # JSON 데이터 처리
from datetime import datetime  # 날짜 및 시간 처리
from collections import deque  # 고정 크기 큐
import tensorflow as tf  # 딥러닝 라이브러리
from tensorflow.keras.models import Sequential, load_model  # 신경망 모델
from tensorflow.keras.layers import Dense, Conv2D, Flatten  # 모델 레이어
from tensorflow.keras.optimizers import Adam  # 최적화기
from tensorflow.keras.callbacks import ModelCheckpoint  # 모델 체크포인트 저장

# 설정 클래스 정의
class Config:
    def __init__(self):
        # 환경 관련 설정
        self.BOARD_WIDTH = 10  # 보드 가로 크기
        self.BOARD_HEIGHT = 20  # 보드 세로 크기
        self.BLOCK_SIZE = 30  # 블록 크기 (픽셀 단위)
        
        # 학습 관련 설정
        self.BATCH_SIZE = 32  # 배치 크기
        self.GAMMA = 0.99  # 할인율
        self.EPSILON_START = 1.0  # 초기 탐험율
        self.EPSILON_MIN = 0.1  # 최소 탐험율
        self.EPSILON_DECAY = 0.999  # 탐험율 감소율
        self.LEARNING_RATE = 0.001  # 학습률
        self.MEMORY_SIZE = 10000  # 경험 메모리 크기
        self.TARGET_UPDATE_FREQUENCY = 100  # 타겟 네트워크 업데이트 주기
        self.SAVE_INTERVAL = 50  # 모델 저장 주기
        self.MAX_STEPS_PER_EPISODE = 500  # 에피소드당 최대 스텝
        
        # 파일 경로 설정
        self.MODEL_DIR = "saved_models"  # 모델 저장 경로
        self.MEMORY_DIR = "saved_memory"  # 메모리 저장 경로
        self.CONFIG_DIR = "config"  # 설정 저장 경로
        
        # 시각화 설정
        self.FPS = 30  # 초당 프레임
        self.COLORS = {
            'BLACK': (0, 0, 0),  # 배경색
            'WHITE': (255, 255, 255),  # 기본 텍스트
            'CYAN': (0, 255, 255)  # 테트리스 블록
        }
        
        # 필요한 디렉토리 생성
        self._create_directories()
        
    def _create_directories(self):
        """필요한 디렉토리를 생성"""
        directories = [self.MODEL_DIR, self.MEMORY_DIR, self.CONFIG_DIR]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            
    def save(self):
        """현재 설정을 JSON 파일로 저장"""
        config_path = os.path.join(self.CONFIG_DIR, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)
            
    def load(self):
        """JSON 파일에서 설정을 로드"""
        config_path = os.path.join(self.CONFIG_DIR, 'config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                loaded_config = json.load(f)
                self.__dict__.update(loaded_config)

# 테트리스 환경 클래스 정의
class TetrisEnv(gym.Env):
    def __init__(self, config):
        super(TetrisEnv, self).__init__()
        self.config = config
        self.width = config.BOARD_WIDTH
        self.height = config.BOARD_HEIGHT
        self.board = np.zeros((self.height, self.width))  # 초기 보드 상태
        
        # 테트리스 블록 모양 정의
        self.shapes = {
            'I': [[1, 1, 1, 1]],  # 막대형 블록
            'O': [[1, 1],
                  [1, 1]],  # 정사각형 블록
            'T': [[0, 1, 0],
                  [1, 1, 1]],  # T자 블록
            'S': [[0, 1, 1],
                  [1, 1, 0]],  # S자 블록
            'Z': [[1, 1, 0],
                  [0, 1, 1]],  # Z자 블록
            'J': [[1, 0, 0],
                  [1, 1, 1]],  # J자 블록
            'L': [[0, 0, 1],
                  [1, 1, 1]]  # L자 블록
        }
        
        # Gym 행동 및 관찰 공간 정의
        self.action_space = gym.spaces.Discrete(4)  # 행동: 왼쪽, 오른쪽, 회전, 드롭
        self.observation_space = gym.spaces.Box(
            low=0, high=1,
            shape=(self.height, self.width),  # 보드 상태
            dtype=np.float32
        )
        
        # 현재 테트리미노 초기화
        self.current_piece = None
        self.piece_position = None
        
    def reset(self):
        """환경 초기화"""
        self.board = np.zeros((self.height, self.width))  # 보드를 초기화
        self._spawn_piece()  # 새로운 테트리미노 생성
        return self._get_state()  # 현재 상태 반환
        
    def _spawn_piece(self):
        """새로운 테트리미노를 보드에 스폰"""
        self.current_piece = np.array(random.choice(list(self.shapes.values())))  # 랜덤 블록 선택
        self.piece_position = [
            0,  # 블록의 초기 y 위치
            self.width // 2 - len(self.current_piece[0]) // 2  # 보드 중앙에 배치
        ]

    def step(self, action):
        """행동을 수행하고 보상, 상태 및 종료 여부 반환"""
        reward = 0  # 기본 보상
        
        if action == 0:  # 왼쪽 이동
            self._move(-1)
        elif action == 1:  # 오른쪽 이동
            self._move(1)
        elif action == 2:  # 회전
            self._rotate()
        else:  # 드롭
            reward = self._drop()  # 드롭 보상
            
        done = self._check_game_over()  # 게임 종료 여부 확인
        next_state = self._get_state()  # 다음 상태 반환
        
        return next_state, reward, done, {}  # Gym에서 필요한 반환 형식
    def _get_state(self):
        """현재 게임 상태 반환"""
        # 현재 보드를 복사
        state = self.board.copy()
        
        # 현재 테트리미노가 있다면 보드에 추가
        if self.current_piece is not None:
            y, x = self.piece_position
            h, w = self.current_piece.shape
            # 테트리미노가 보드 범위 내에 위치할 경우 추가
            if y >= 0 and x >= 0 and y + h <= self.height and x + w <= self.width:
                state[y:y+h, x:x+w] += self.current_piece
        
        return state

    def _move(self, dx):
        """테트리미노를 왼쪽 또는 오른쪽으로 이동"""
        old_x = self.piece_position[1]
        new_x = old_x + dx
        
        # 이동 가능한 경우 위치 업데이트
        if self._is_valid_move(self.piece_position[0], new_x):
            self.piece_position[1] = new_x
            return True
        return False

    def _rotate(self):
        """테트리미노 회전"""
        rotated_piece = np.rot90(self.current_piece)  # 시계 방향으로 회전
        # 회전 후에도 유효한 위치라면 회전을 적용
        if self._is_valid_move(self.piece_position[0], self.piece_position[1], rotated_piece):
            self.current_piece = rotated_piece
            return True
        return False

    def _drop(self):
        """테트리미노를 한 번에 보드 바닥으로 이동"""
        reward = 0
        # 가능한 한 많이 아래로 이동
        while self._move_down():
            reward += 1  # 이동할 때마다 보상 증가
        
        # 테트리미노를 보드에 고정하고 줄 제거 및 보상 계산
        self._place_piece()
        lines_cleared = self._clear_lines()
        reward += self._calculate_line_clear_reward(lines_cleared)
        # 새로운 테트리미노 생성
        self._spawn_piece()
        
        return reward

    def _move_down(self):
        """테트리미노를 한 칸 아래로 이동"""
        # 유효한 아래 이동이 가능한 경우 위치 업데이트
        if self._is_valid_move(self.piece_position[0] + 1, self.piece_position[1]):
            self.piece_position[0] += 1
            return True
        return False

    def _is_valid_move(self, new_y, new_x, piece=None):
        """테트리미노의 새로운 위치 유효성 검사"""
        if piece is None:
            piece = self.current_piece  # 현재 테트리미노 사용
            
        h, w = piece.shape
        
        # 경계 조건 검사
        if new_x < 0 or new_x + w > self.width:
            return False
        if new_y + h > self.height:
            return False
            
        # 다른 블록과 충돌하는지 검사
        for y in range(h):
            for x in range(w):
                if piece[y][x]:
                    board_y = new_y + y
                    board_x = new_x + x
                    if board_y >= 0 and self.board[board_y][board_x]:
                        return False
        return True

    def _place_piece(self):
        """현재 테트리미노를 보드에 고정"""
        y, x = self.piece_position
        h, w = self.current_piece.shape
        
        # 테트리미노의 모든 블록을 보드에 추가
        for i in range(h):
            for j in range(w):
                if self.current_piece[i][j]:
                    self.board[y+i][x+j] = 1

    def _clear_lines(self):
        """완성된 줄을 제거하고 개수를 반환"""
        lines_to_clear = []
        # 모든 줄을 검사하여 완전히 채워진 줄을 찾아 제거
        for i in range(self.height):
            if np.all(self.board[i]):
                lines_to_clear.append(i)
                
        # 완성된 줄 제거 후 위의 블록을 아래로 이동
        for line in lines_to_clear:
            self.board = np.vstack((np.zeros((1, self.width)), self.board[:line]))
            
        return len(lines_to_clear)

    def _calculate_line_clear_reward(self, lines_cleared):
        """줄 제거 보상을 계산"""
        if lines_cleared == 0:
            return 0
        elif lines_cleared == 1:
            return 100  # 1줄 제거 보상
        elif lines_cleared == 2:
            return 300  # 2줄 제거 보상
        elif lines_cleared == 3:
            return 500  # 3줄 제거 보상
        else:
            return 800  # 4줄 제거 보상 (테트리스)

    def _check_game_over(self):
        """게임 종료 조건 확인"""
        # 첫 번째 행에 블록이 존재하면 게임 종료
        return np.any(self.board[0])

# 상태 처리기 클래스 정의
class StateProcessor:
    def __init__(self, config):
        """상태 처리기 초기화"""
        self.config = config
        self.height = config.BOARD_HEIGHT
        self.width = config.BOARD_WIDTH

    def process_state(self, state):
        """상태를 처리하고 특징을 추출"""
        processed_state = state.astype(np.float32)  # 상태를 float32로 변환
        heights = self._get_heights(processed_state)  # 각 열의 높이 계산
        
        # 다양한 특징 계산
        features = {
            'holes': self._count_holes(processed_state, heights),  # 구멍의 개수
            'bumpiness': self._calculate_bumpiness(heights),  # 울퉁불퉁함
            'height': np.max(heights),  # 최대 높이
            'avg_height': np.mean(heights),  # 평균 높이
            'completed_lines': self._count_completed_lines(processed_state),  # 완성된 줄 개수
            'wells': self._calculate_wells(heights),  # 우물 깊이
            'row_transitions': self._count_row_transitions(processed_state),  # 행 전환점
            'col_transitions': self._count_col_transitions(processed_state),  # 열 전환점
            'pit_depth': self._calculate_pit_depth(heights),  # 구덩이 깊이
            'structure_score': self._evaluate_structure(processed_state, heights)  # 구조 점수
        }
        
        return processed_state, features

    def _get_heights(self, board):
        """각 열의 높이를 계산"""
        heights = np.zeros(self.width)
        for col in range(self.width):
            for row in range(self.height):
                if board[row][col]:  # 블록이 있는 첫 번째 위치를 찾으면 높이 저장
                    heights[col] = self.height - row
                    break
        return heights

    def _count_holes(self, board, heights):
        """각 열에서 구멍(빈 공간) 개수 계산"""
        holes = 0
        for col in range(self.width):
            # 해당 열에서 블록이 있는 첫 행 아래로 검사
            if heights[col] > 0:  # 블록이 존재하는 열만 확인
                for row in range(self.height - int(heights[col]), self.height):
                    if board[row][col] == 0:  # 빈 공간 확인
                        holes += 1
        return holes

    def _calculate_bumpiness(self, heights):
        """열 높이 간의 울퉁불퉁함(차이값 합) 계산"""
        return np.sum(np.abs(np.diff(heights)))  # 높이 차이의 절댓값 합산

    def _count_completed_lines(self, board):
        """완성된 줄의 개수를 계산"""
        return np.sum([np.all(row) for row in board])  # 모든 열이 채워진 행 개수 반환

    def _calculate_wells(self, heights):
        """우물(깊은 홈)의 개수와 깊이 계산"""
        wells = 0
        
        # 첫 열과 마지막 열 처리
        if heights[0] < heights[1]:
            wells += heights[1] - heights[0]
        if heights[-1] < heights[-2]:
            wells += heights[-2] - heights[-1]
        
        # 중간 열 처리
        for i in range(1, self.width - 1):
            left = heights[i - 1]
            right = heights[i + 1]
            current = heights[i]
            if current < left and current < right:  # 좌우보다 낮으면 우물로 간주
                wells += min(left, right) - current
        return wells

    def _count_row_transitions(self, board):
        """가로 방향 전환점(0에서 1 또는 1에서 0으로 바뀌는 지점) 개수 계산"""
        transitions = 0
        for row in range(self.height):
            for col in range(self.width - 1):
                if board[row][col] != board[row][col + 1]:
                    transitions += 1
        return transitions

    def _count_col_transitions(self, board):
        """세로 방향 전환점(0에서 1 또는 1에서 0으로 바뀌는 지점) 개수 계산"""
        transitions = 0
        for col in range(self.width):
            for row in range(self.height - 1):
                if board[row][col] != board[row + 1][col]:
                    transitions += 1
        return transitions

    def _calculate_pit_depth(self, heights):
        """구덩이(양 옆보다 낮은 열)의 깊이 계산"""
        pit_depth = 0
        for col in range(self.width):
            # 양쪽 열이 존재하는 중앙 열에서 구덩이 깊이를 계산
            if col > 0 and col < self.width - 1:
                if heights[col] < heights[col - 1] - 2 and heights[col] < heights[col + 1] - 2:
                    pit_depth += min(heights[col - 1], heights[col + 1]) - heights[col]
        return pit_depth

    def _evaluate_structure(self, board, heights):
        """보드의 전체 구조 점수 계산"""
        structure_score = 0
        
        # 높이 차이에 따른 페널티
        height_diff_penalty = self._calculate_bumpiness(heights) * -0.5
        
        # 낮은 평균 높이에 대한 보너스
        low_height_bonus = (20 - np.mean(heights)) * 2
        
        # 구멍의 개수에 따른 페널티
        holes_penalty = self._count_holes(board, heights) * -2
        
        # 가장자리 높이에 따른 보너스 (테트리스 공간 확보)
        edge_bonus = (heights[0] + heights[-1]) * 0.5
        
        # 구조 점수의 최종 계산
        structure_score = height_diff_penalty + low_height_bonus + holes_penalty + edge_bonus
        return structure_score

# 보상 시스템 클래스
class RewardSystem:
    def __init__(self, config):
        """보상 시스템 초기화"""
        self.config = config
        self.weights = {
            'lines_cleared': 500,  # 줄 제거 보상
            'holes': -50,  # 구멍 페널티
            'bumpiness': -20,  # 울퉁불퉁함 페널티
            'height': -30,  # 최대 높이 페널티
            'wells': -15,  # 우물 깊이 페널티
            'transitions': -10,  # 전환점 페널티
            'pit_depth': -25,  # 구덩이 깊이 페널티
            'structure': 40,  # 보드 구조 보상
        }
        self.last_lines_cleared = 0  # 이전 에피소드에서 제거한 줄 개수 기록

    def calculate_reward(self, features, lines_cleared):
        """보상 계산"""
        reward = 0
        
        # 줄 제거 보상
        reward += lines_cleared * self.weights['lines_cleared']
        
        # 구멍, 울퉁불퉁함, 높이 등의 페널티
        reward += features['holes'] * self.weights['holes']
        reward += features['bumpiness'] * self.weights['bumpiness']
        reward += features['height'] * self.weights['height']
        reward += features['wells'] * self.weights['wells']
        reward += (features['row_transitions'] + features['col_transitions']) * self.weights['transitions']
        reward += features['pit_depth'] * self.weights['pit_depth']
        reward += features['structure_score'] * self.weights['structure']
        
        # 연속 줄 제거 보너스
        if lines_cleared > 0 and self.last_lines_cleared > 0:
            reward += 200  # 연속 보너스
        
        # 게임 오버 페널티
        if features.get('game_over', False):
            reward -= 1000  # 게임 오버 페널티
        
        # 마지막 줄 제거 기록 갱신
        self.last_lines_cleared = lines_cleared
        return reward

# 메모리 관리 클래스
class MemoryManager:
    def __init__(self, config):
        """메모리 관리 초기화"""
        self.config = config
        self.memory_path = config.MEMORY_DIR
        self.max_memory_files = 5  # 최대 저장할 메모리 파일 수

    def save_memory(self, memory, episode):
        """경험 메모리를 파일로 저장"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"memory_ep{episode}_{timestamp}.pkl"
        filepath = os.path.join(self.memory_path, filename)
        
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(list(memory), f)  # 메모리를 파일에 직렬화하여 저장
            
            # 오래된 메모리 파일 삭제
            self._cleanup_old_files()
            return True
        except Exception as e:
            print(f"메모리 저장 실패: {e}")
            return False

    def load_latest_memory(self):
        """가장 최근에 저장된 메모리 파일을 불러오기"""
        memory_files = glob.glob(os.path.join(self.memory_path, "memory_*.pkl"))
        if not memory_files:
            return None  # 저장된 메모리가 없으면 None 반환
            
        latest_file = max(memory_files, key=os.path.getctime)  # 최신 파일 찾기
        try:
            with open(latest_file, 'rb') as f:
                return pickle.load(f)  # 파일에서 메모리를 역직렬화하여 로드
        except Exception as e:
            print(f"메모리 로드 실패: {e}")
            return None

    def _cleanup_old_files(self):
        """오래된 메모리 파일 삭제"""
        memory_files = glob.glob(os.path.join(self.memory_path, "memory_*.pkl"))
        if len(memory_files) > self.max_memory_files:
            memory_files.sort(key=os.path.getctime)  # 생성 날짜 순으로 정렬
            for file in memory_files[:-self.max_memory_files]:  # 가장 오래된 파일부터 삭제
                try:
                    os.remove(file)
                except Exception as e:
                    print(f"파일 삭제 실패: {e}")

# DQN 에이전트 클래스
class DQNAgent:
    def __init__(self, state_size, action_size, config):
        """DQN 에이전트 초기화"""
        self.state_size = state_size  # 상태 공간 크기
        self.action_size = action_size  # 행동 공간 크기
        self.config = config
        self.memory = deque(maxlen=config.MEMORY_SIZE)  # 경험 메모리
        self.gamma = config.GAMMA  # 할인율
        self.epsilon = config.EPSILON_START  # 초기 탐험율
        self.epsilon_min = config.EPSILON_MIN  # 최소 탐험율
        self.epsilon_decay = config.EPSILON_DECAY  # 탐험율 감소율
        self.learning_rate = config.LEARNING_RATE  # 학습률
        self.batch_size = config.BATCH_SIZE  # 배치 크기
        self.train_start = self.batch_size  # 학습 시작 조건
        
        # 모델 생성
        self.model = self._build_model()  # 정책 네트워크
        self.target_model = self._build_model()  # 타겟 네트워크
        self.update_target_model()  # 타겟 네트워크 초기화
        
        # 메모리 관리자 초기화
        self.memory_manager = MemoryManager(config)
        self._load_memory()  # 저장된 메모리 로드

    def _build_model(self):
        """신경망 모델 생성"""
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', padding='same', 
                   input_shape=(self.state_size[0], self.state_size[1], 1)),  # 입력 크기
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            Flatten(),  # 평탄화하여 완전 연결 레이어로 입력
            Dense(256, activation='relu'),  # 은닉층
            Dense(self.action_size, activation='linear')  # 출력층 (Q값)
        ])
        model.compile(loss=tf.keras.losses.Huber(), optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        """정책 네트워크의 가중치를 타겟 네트워크로 복사"""
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        """경험 메모리에 저장"""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """현재 상태에서 행동 선택"""
        if np.random.rand() <= self.epsilon:  # 랜덤 탐험
            return random.randrange(self.action_size)  # 임의의 행동 반환
        
        # 정책 네트워크를 사용하여 Q값 예측
        state = np.reshape(state, [1, self.state_size[0], self.state_size[1], 1])
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])  # 가장 큰 Q값을 가지는 행동 선택

    def replay(self, batch_size):
        """경험 재생으로 모델 학습"""
        if len(self.memory) < self.train_start:  # 충분한 메모리가 없으면 학습 생략
            return
            
        minibatch = random.sample(self.memory, batch_size)  # 랜덤 샘플링
        states = np.zeros((batch_size, self.state_size[0], self.state_size[1], 1))
        next_states = np.zeros((batch_size, self.state_size[0], self.state_size[1], 1))
        actions, rewards, dones = [], [], []

        # 미니배치 구성
        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            states[i] = np.reshape(state, [self.state_size[0], self.state_size[1], 1])
            next_states[i] = np.reshape(next_state, [self.state_size[0], self.state_size[1], 1])
            actions.append(action)
            rewards.append(reward)
            dones.append(done)

        # 현재 및 다음 상태의 Q값 예측
        target = self.model.predict(states, verbose=0)
        target_next = self.target_model.predict(next_states, verbose=0)

        # 타겟값 갱신
        for i in range(batch_size):
            if dones[i]:
                target[i][actions[i]] = rewards[i]  # 종료 상태에서는 보상만 고려
            else:
                # Q(s, a) = r + γ * max Q'(s', a')
                target[i][actions[i]] = rewards[i] + self.gamma * np.amax(target_next[i])

        # 정책 네트워크 업데이트
        self.model.fit(states, target, batch_size=batch_size, epochs=1, verbose=0)

        # 탐험율 감소
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_model(self, episode):
        """모델과 상태를 저장"""
        try:
            model_path = os.path.join(self.config.MODEL_DIR, f"tetris_model_ep{episode}.keras")
            self.model.save(model_path)
            
            # 탐험율과 에피소드 상태 저장
            state_path = os.path.join(self.config.MODEL_DIR, f"model_state_ep{episode}.json")
            state_data = {
                'epsilon': float(self.epsilon),
                'episode': int(episode)
            }
            with open(state_path, 'w') as f:
                json.dump(state_data, f)
                
            print(f"모델 저장 완료: {model_path}")
        except Exception as e:
            print(f"모델 저장 실패: {e}")

    def load_model(self):
        """최신 저장된 모델 불러오기"""
        try:
            model_files = glob.glob(os.path.join(self.config.MODEL_DIR, "tetris_model_*.keras"))
            if not model_files:
                print("저장된 모델이 없습니다.")
                return False
                
            latest_model = max(model_files, key=os.path.getctime)  # 최신 모델 찾기
            state_file = latest_model.replace('tetris_model', 'model_state').replace('.keras', '.json')
            
            # 모델 로드
            self.model = tf.keras.models.load_model(latest_model)
            self.target_model = tf.keras.models.load_model(latest_model)
            
            # 상태 로드
            if os.path.exists(state_file):
                with open(state_file, 'r') as f:
                    state_data = json.load(f)
                    self.epsilon = float(state_data['epsilon'])
                    
            print(f"모델 로드 완료: {latest_model}")
            return True
            
        except Exception as e:
            print(f"모델 로드 실패: {e}")
            return False
            
    def _load_memory(self):
        """저장된 메모리를 로드"""
        try:
            loaded_memory = self.memory_manager.load_latest_memory()
            if loaded_memory:
                self.memory = deque(loaded_memory, maxlen=self.config.MEMORY_SIZE)
                print(f"메모리 로드 완료: {len(self.memory)} 개의 경험")
        except Exception as e:
            print(f"메모리 로드 실패: {e}")

    def save_memory(self, episode):
        """현재 메모리를 저장"""
        try:
            self.memory_manager.save_memory(self.memory, episode)
        except Exception as e:
            print(f"메모리 저장 실패: {e}")

# GUI 클래스
class TetrisGUI:
    def __init__(self, config):
        """GUI 초기화"""
        pygame.init()  # Pygame 초기화
        self.config = config
        self.block_size = config.BLOCK_SIZE  # 블록 크기 설정
        self.width = config.BOARD_WIDTH * self.block_size  # 보드의 가로 크기 (픽셀)
        self.height = config.BOARD_HEIGHT * self.block_size  # 보드의 세로 크기 (픽셀)
        
        # Pygame 창 설정
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Tetris AI')  # 창 제목 설정
        
        # 프레임 속도 조절용 시계
        self.clock = pygame.time.Clock()

    def draw(self, state):
        """현재 상태를 화면에 그리기"""
        self.screen.fill(self.config.COLORS['BLACK'])  # 배경색 채우기
        
        # 상태(board)에 있는 블록을 화면에 그리기
        for y in range(len(state)):
            for x in range(len(state[0])):
                if state[y][x]:  # 블록이 있는 위치만 그리기
                    pygame.draw.rect(
                        self.screen,
                        self.config.COLORS['CYAN'],  # 블록 색상
                        [x * self.block_size,  # 왼쪽 위 x좌표
                         y * self.block_size,  # 왼쪽 위 y좌표
                         self.block_size - 1,  # 너비 (테두리 간격 포함)
                         self.block_size - 1]  # 높이 (테두리 간격 포함)
                    )
        
        # 화면 업데이트
        pygame.display.flip()
        self.clock.tick(self.config.FPS)  # FPS에 맞춰 속도 조절

# 학습 관리 클래스
class TrainingManager:
    def __init__(self, config):
        """학습 결과 로깅 및 관리"""
        self.config = config
        self.best_score = float('-inf')  # 최고 점수 초기화
        self.scores_history = []  # 점수 기록
        self.lines_history = []  # 줄 제거 기록

    def log_episode(self, episode, score, lines_cleared, steps):
        """에피소드 결과 로깅 및 최고 점수 갱신 확인"""
        self.scores_history.append(score)  # 에피소드 점수 추가
        self.lines_history.append(lines_cleared)  # 제거한 줄 수 추가
        
        # 최근 100개 에피소드의 평균 계산
        avg_score = np.mean(self.scores_history[-100:])
        avg_lines = np.mean(self.lines_history[-100:])
        
        # 결과 출력
        print(f"에피소드: {episode}")
        print(f"점수: {score:.2f} (평균: {avg_score:.2f})")
        print(f"제거한 줄: {lines_cleared} (평균: {avg_lines:.2f})")
        print(f"진행 스텝: {steps}")
        print("-" * 50)
        
        # 최고 점수 갱신 확인
        if score > self.best_score:
            self.best_score = score  # 최고 점수 업데이트
            return True  # 최고 점수 갱신 여부 반환
        return False

# 메인 학습 루프
def main():
    """테트리스 AI 학습 메인 루프"""
    # 설정 로드
    config = Config()
    config.load()  # 기존 설정이 있으면 로드
    
    # 환경, 에이전트 및 기타 구성요소 초기화
    env = TetrisEnv(config)
    state_size = (config.BOARD_HEIGHT, config.BOARD_WIDTH)
    action_size = env.action_space.n
    
    agent = DQNAgent(state_size, action_size, config)  # DQN 에이전트 초기화
    state_processor = StateProcessor(config)  # 상태 처리기 초기화
    reward_system = RewardSystem(config)  # 보상 시스템 초기화
    gui = TetrisGUI(config)  # GUI 초기화
    training_manager = TrainingManager(config)  # 학습 관리 초기화
    
    # 이전에 저장된 모델 및 메모리 로드
    agent.load_model()
    
    episodes = 1000  # 학습 에피소드 수
    try:
        for episode in range(episodes):
            state = env.reset()  # 환경 초기화
            total_reward = 0  # 에피소드 총 보상
            steps = 0  # 스텝 수 초기화
            lines_cleared = 0  # 제거된 줄 수 초기화
            
            while steps < config.MAX_STEPS_PER_EPISODE:
                # GUI 업데이트
                gui.draw(state)
                
                # 행동 선택 및 환경 단계 진행
                action = agent.act(state)  # 행동 선택
                next_state, reward, done, _ = env.step(action)  # 환경 업데이트
                
                # 상태 처리 및 보상 계산
                processed_state, features = state_processor.process_state(next_state)
                features['game_over'] = done  # 게임 종료 상태 추가
                reward = reward_system.calculate_reward(features, features['completed_lines'])
                
                # 에이전트에 경험 저장
                agent.remember(state, action, reward, next_state, done)
                
                # 경험 재생(학습)
                if len(agent.memory) >= agent.batch_size:
                    agent.replay(agent.batch_size)
                
                # 상태 및 기타 변수 업데이트
                state = next_state
                total_reward += reward
                steps += 1
                lines_cleared += features['completed_lines']
                
                # 주기적으로 가비지 컬렉션 수행
                if steps % 100 == 0:
                    gc.collect()
                
                # 게임 종료 처리
                if done:
                    break
                
                # Pygame 이벤트 처리
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:  # 창 닫기 이벤트
                        # 현재 모델과 메모리 저장 후 종료
                        agent.save_model(episode)
                        agent.save_memory(episode)
                        pygame.quit()
                        return
            
            # 에피소드 결과 기록
            is_best = training_manager.log_episode(episode, total_reward, lines_cleared, steps)
            
            # 주기적으로 모델 및 메모리 저장
            if (episode + 1) % config.SAVE_INTERVAL == 0 or is_best:
                agent.save_model(episode)
                agent.save_memory(episode)
            
            # 타겟 네트워크 업데이트
            if episode % config.TARGET_UPDATE_FREQUENCY == 0:
                agent.update_target_model()
                
    except KeyboardInterrupt:
        print("\n학습 중단...")  # 키보드 인터럽트 처리
    except Exception as e:
        print(f"오류 발생: {e}")
    finally:
        # 종료 시 모델 및 메모리 저장
        agent.save_model(episode)
        agent.save_memory(episode)
        pygame.quit()
        
    # 설정 저장
    config.save()

# 프로그램 실행
if __name__ == "__main__":
    # 초기화 및 메모리 정리
    gc.enable()
    tf.keras.backend.clear_session()
    
    try:
        main()  # 메인 함수 실행
    except Exception as e:
        print(f"프로그램 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 프로그램 종료 시 메모리 정리
        pygame.quit()
        gc.collect()
