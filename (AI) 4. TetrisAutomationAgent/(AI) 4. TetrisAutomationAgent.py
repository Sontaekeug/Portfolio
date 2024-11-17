import gym
import pygame
import numpy as np
import random
import os
import gc
import time
import glob
import pickle
import json
from datetime import datetime
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

# 설정 클래스 추가
class Config:
    def __init__(self):
        # 환경 설정
        self.BOARD_WIDTH = 10
        self.BOARD_HEIGHT = 20
        self.BLOCK_SIZE = 30
        
        # 학습 설정
        self.BATCH_SIZE = 32
        self.GAMMA = 0.99
        self.EPSILON_START = 1.0
        self.EPSILON_MIN = 0.1
        self.EPSILON_DECAY = 0.999
        self.LEARNING_RATE = 0.001
        self.MEMORY_SIZE = 10000
        self.TARGET_UPDATE_FREQUENCY = 100
        self.SAVE_INTERVAL = 50
        self.MAX_STEPS_PER_EPISODE = 500
        
        # 파일 경로 설정
        self.MODEL_DIR = "saved_models"
        self.MEMORY_DIR = "saved_memory"
        self.CONFIG_DIR = "config"
        
        # 시각화 설정
        self.FPS = 30
        self.COLORS = {
            'BLACK': (0, 0, 0),
            'WHITE': (255, 255, 255),
            'CYAN': (0, 255, 255)
        }
        
        # 생성 시 필요한 디렉토리 생성
        self._create_directories()
        
    def _create_directories(self):
        """필요한 디렉토리 생성"""
        directories = [self.MODEL_DIR, self.MEMORY_DIR, self.CONFIG_DIR]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            
    def save(self):
        """설정 저장"""
        config_path = os.path.join(self.CONFIG_DIR, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)
            
    def load(self):
        """설정 로드"""
        config_path = os.path.join(self.CONFIG_DIR, 'config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                loaded_config = json.load(f)
                self.__dict__.update(loaded_config)

# 테트리스 환경 클래스
class TetrisEnv(gym.Env):
    def __init__(self, config):
        super(TetrisEnv, self).__init__()
        
        self.config = config
        self.width = config.BOARD_WIDTH
        self.height = config.BOARD_HEIGHT
        self.board = np.zeros((self.height, self.width))
        
        # 테트리미노 모양 정의
        self.shapes = {
            'I': [[1, 1, 1, 1]],
            'O': [[1, 1],
                  [1, 1]],
            'T': [[0, 1, 0],
                  [1, 1, 1]],
            'S': [[0, 1, 1],
                  [1, 1, 0]],
            'Z': [[1, 1, 0],
                  [0, 1, 1]],
            'J': [[1, 0, 0],
                  [1, 1, 1]],
            'L': [[0, 0, 1],
                  [1, 1, 1]]
        }
        
        # 행동 공간과 관찰 공간 정의
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(
            low=0, high=1,
            shape=(self.height, self.width),
            dtype=np.float32
        )
        
        # 현재 테트리미노 초기화
        self.current_piece = None
        self.piece_position = None
        
    def reset(self):
        """환경 초기화"""
        self.board = np.zeros((self.height, self.width))
        self._spawn_piece()
        return self._get_state()
        
    def _spawn_piece(self):
        """새로운 테트리미노 생성"""
        self.current_piece = np.array(random.choice(list(self.shapes.values())))
        self.piece_position = [
            0,
            self.width // 2 - len(self.current_piece[0]) // 2
        ]
    def _get_state(self):
        """현재 게임 상태 반환"""
        state = self.board.copy()
        
        if self.current_piece is not None:
            y, x = self.piece_position
            h, w = self.current_piece.shape
            # 보드 범위 내에서만 현재 조각 표시
            if y >= 0 and x >= 0 and y + h <= self.height and x + w <= self.width:
                state[y:y+h, x:x+w] += self.current_piece
        
        return state
    
    def step(self, action):
        """행동 실행 및 결과 반환"""
        reward = 0
        
        if action == 0:  # 왼쪽
            self._move(-1)
        elif action == 1:  # 오른쪽
            self._move(1)
        elif action == 2:  # 회전
            self._rotate()
        else:  # 드롭
            reward = self._drop()
            
        done = self._check_game_over()
        next_state = self._get_state()
        
        return next_state, reward, done, {}
    
    def _move(self, dx):
        """테트리미노 이동"""
        old_x = self.piece_position[1]
        new_x = old_x + dx
        
        if self._is_valid_move(self.piece_position[0], new_x):
            self.piece_position[1] = new_x
            return True
        return False
    
    def _rotate(self):
        """테트리미노 회전"""
        rotated_piece = np.rot90(self.current_piece)
        if self._is_valid_move(self.piece_position[0], self.piece_position[1], rotated_piece):
            self.current_piece = rotated_piece
            return True
        return False
    
    def _drop(self):
        """테트리미노 드롭"""
        reward = 0
        while self._move_down():
            reward += 1
        
        self._place_piece()
        lines_cleared = self._clear_lines()
        reward += self._calculate_line_clear_reward(lines_cleared)
        self._spawn_piece()
        
        return reward
    
    def _move_down(self):
        """테트리미노 한 칸 아래로 이동"""
        if self._is_valid_move(self.piece_position[0] + 1, self.piece_position[1]):
            self.piece_position[0] += 1
            return True
        return False
    
    def _is_valid_move(self, new_y, new_x, piece=None):
        """이동 유효성 검사"""
        if piece is None:
            piece = self.current_piece
            
        h, w = piece.shape
        
        # 경계 체크
        if new_x < 0 or new_x + w > self.width:
            return False
        if new_y + h > self.height:
            return False
            
        # 충돌 체크
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
        
        for i in range(h):
            for j in range(w):
                if self.current_piece[i][j]:
                    self.board[y+i][x+j] = 1
    
    def _clear_lines(self):
        """완성된 줄 제거"""
        lines_to_clear = []
        for i in range(self.height):
            if np.all(self.board[i]):
                lines_to_clear.append(i)
                
        for line in lines_to_clear:
            self.board = np.vstack((np.zeros((1, self.width)), self.board[:line]))
            
        return len(lines_to_clear)
    
    def _calculate_line_clear_reward(self, lines_cleared):
        """줄 제거에 대한 보상 계산"""
        if lines_cleared == 0:
            return 0
        elif lines_cleared == 1:
            return 100
        elif lines_cleared == 2:
            return 300
        elif lines_cleared == 3:
            return 500
        else:
            return 800  # 테트리스
    
    def _check_game_over(self):
        """게임 오버 체크"""
        return np.any(self.board[0])

# 상태 처리기 클래스
class StateProcessor:
    def __init__(self, config):
        self.config = config
        self.height = config.BOARD_HEIGHT
        self.width = config.BOARD_WIDTH

    def process_state(self, state):
        """상태 처리 및 특징 추출"""
        processed_state = state.astype(np.float32)
        heights = self._get_heights(processed_state)
        
        features = {
            'holes': self._count_holes(processed_state, heights),
            'bumpiness': self._calculate_bumpiness(heights),
            'height': np.max(heights),
            'avg_height': np.mean(heights),
            'completed_lines': self._count_completed_lines(processed_state),
            'wells': self._calculate_wells(heights),
            'row_transitions': self._count_row_transitions(processed_state),
            'col_transitions': self._count_col_transitions(processed_state),
            'pit_depth': self._calculate_pit_depth(heights),
            'structure_score': self._evaluate_structure(processed_state, heights)
        }
        
        return processed_state, features

    def _get_heights(self, board):
        """각 열의 높이 계산"""
        heights = np.zeros(self.width)
        for col in range(self.width):
            for row in range(self.height):
                if board[row][col]:
                    heights[col] = self.height - row
                    break
        return heights
    
    def _count_holes(self, board, heights):
        """빈 공간(구멍) 계산"""
        holes = 0
        for col in range(self.width):
            if heights[col] > 0:  # 해당 열에 블록이 있는 경우만 검사
                for row in range(self.height - int(heights[col]), self.height):
                    if board[row][col] == 0:
                        holes += 1
        return holes

    def _calculate_bumpiness(self, heights):
        """표면 울퉁불퉁함 계산"""
        return np.sum(np.abs(np.diff(heights)))

    def _count_completed_lines(self, board):
        """완성된 줄 수 계산"""
        return np.sum([np.all(row) for row in board])

    def _calculate_wells(self, heights):
        """우물 깊이 계산"""
        wells = 0
        # 좌우 끝 열 처리
        if heights[0] < heights[1]:
            wells += heights[1] - heights[0]
        if heights[-1] < heights[-2]:
            wells += heights[-2] - heights[-1]
        
        # 중간 열들 처리
        for i in range(1, self.width-1):
            left = heights[i-1]
            right = heights[i+1]
            current = heights[i]
            if current < left and current < right:
                wells += min(left, right) - current
        return wells

    def _count_row_transitions(self, board):
        """가로 방향 전환점 계산"""
        transitions = 0
        for row in range(self.height):
            for col in range(self.width - 1):
                if board[row][col] != board[row][col + 1]:
                    transitions += 1
        return transitions

    def _count_col_transitions(self, board):
        """세로 방향 전환점 계산"""
        transitions = 0
        for col in range(self.width):
            for row in range(self.height - 1):
                if board[row][col] != board[row + 1][col]:
                    transitions += 1
        return transitions

    def _calculate_pit_depth(self, heights):
        """구덩이 깊이 계산"""
        pit_depth = 0
        for col in range(self.width):
            if col > 0 and col < self.width - 1:
                if heights[col] < heights[col-1] - 2 and heights[col] < heights[col+1] - 2:
                    pit_depth += min(heights[col-1], heights[col+1]) - heights[col]
        return pit_depth

    def _evaluate_structure(self, board, heights):
        """전체 구조 평가"""
        structure_score = 0
        
        # 높이 차이에 대한 페널티
        height_diff_penalty = self._calculate_bumpiness(heights) * -0.5
        
        # 낮은 높이 보너스
        low_height_bonus = (20 - np.mean(heights)) * 2
        
        # 구멍 근처 블록에 대한 페널티
        holes_penalty = self._count_holes(board, heights) * -2
        
        # 가장자리 높이 보너스 (테트리스 공간 확보)
        edge_bonus = (heights[0] + heights[-1]) * 0.5
        
        structure_score = height_diff_penalty + low_height_bonus + holes_penalty + edge_bonus
        return structure_score

# 보상 시스템 클래스
class RewardSystem:
    def __init__(self, config):
        self.config = config
        self.weights = {
            'lines_cleared': 500,      # 줄 제거 보상
            'holes': -50,              # 구멍 페널티
            'bumpiness': -20,          # 울퉁불퉁함 페널티
            'height': -30,             # 높이 페널티
            'wells': -15,              # 우물 페널티
            'transitions': -10,        # 전환점 페널티
            'pit_depth': -25,          # 구덩이 페널티
            'structure': 40,           # 구조 보상
        }
        self.last_lines_cleared = 0

    def calculate_reward(self, features, lines_cleared):
        """보상 계산"""
        reward = 0
        
        # 기본 보상 계산
        reward += lines_cleared * self.weights['lines_cleared']
        reward += features['holes'] * self.weights['holes']
        reward += features['bumpiness'] * self.weights['bumpiness']
        reward += features['height'] * self.weights['height']
        reward += features['wells'] * self.weights['wells']
        reward += (features['row_transitions'] + 
                  features['col_transitions']) * self.weights['transitions']
        reward += features['pit_depth'] * self.weights['pit_depth']
        reward += features['structure_score'] * self.weights['structure']
        
        # 연속 줄 제거 보너스
        if lines_cleared > 0 and self.last_lines_cleared > 0:
            reward += 200  # 연속 보너스
        
        # 게임 오버 페널티
        if features.get('game_over', False):
            reward -= 1000
        
        self.last_lines_cleared = lines_cleared
        return reward

# 메모리 관리 클래스 추가
class MemoryManager:
    def __init__(self, config):
        self.config = config
        self.memory_path = config.MEMORY_DIR
        self.max_memory_files = 5  # 최대 메모리 파일 수

    def save_memory(self, memory, episode):
        """경험 메모리 저장"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"memory_ep{episode}_{timestamp}.pkl"
        filepath = os.path.join(self.memory_path, filename)
        
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(list(memory), f)
            
            # 오래된 메모리 파일 정리
            self._cleanup_old_files()
            
            return True
        except Exception as e:
            print(f"메모리 저장 실패: {e}")
            return False

    def load_latest_memory(self):
        """최신 메모리 불러오기"""
        memory_files = glob.glob(os.path.join(self.memory_path, "memory_*.pkl"))
        if not memory_files:
            return None
            
        latest_file = max(memory_files, key=os.path.getctime)
        try:
            with open(latest_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"메모리 로드 실패: {e}")
            return None

    def _cleanup_old_files(self):
        """오래된 메모리 파일 정리"""
        memory_files = glob.glob(os.path.join(self.memory_path, "memory_*.pkl"))
        if len(memory_files) > self.max_memory_files:
            memory_files.sort(key=os.path.getctime)
            for file in memory_files[:-self.max_memory_files]:
                try:
                    os.remove(file)
                except Exception as e:
                    print(f"파일 삭제 실패: {e}")
    
# DQN 에이전트 클래스
class DQNAgent:
    def __init__(self, state_size, action_size, config):
        self.state_size = state_size
        self.action_size = action_size
        self.config = config
        self.memory = deque(maxlen=config.MEMORY_SIZE)
        self.gamma = config.GAMMA
        self.epsilon = config.EPSILON_START
        self.epsilon_min = config.EPSILON_MIN
        self.epsilon_decay = config.EPSILON_DECAY
        self.learning_rate = config.LEARNING_RATE
        self.batch_size = config.BATCH_SIZE
        self.train_start = self.batch_size
        
        # 모델 초기화
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        
        # 메모리 관리자 초기화
        self.memory_manager = MemoryManager(config)
        self._load_memory()
        
    def _build_model(self):
        """신경망 모델 구축"""
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', padding='same',
                  input_shape=(self.state_size[0], self.state_size[1], 1)),
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            Flatten(),
            Dense(256, activation='relu'),
            Dense(self.action_size, activation='linear')
        ])
        # Huber Loss를 직접 정의
        model.compile(loss=tf.keras.losses.Huber(),
                     optimizer=Adam(learning_rate=self.learning_rate))
        return model
        
    def update_target_model(self):
        """타겟 모델 업데이트"""
        self.target_model.set_weights(self.model.get_weights())
        
    def remember(self, state, action, reward, next_state, done):
        """경험 저장"""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """행동 선택"""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state = np.reshape(state, [1, self.state_size[0], self.state_size[1], 1])
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        """경험 재생을 통한 학습"""
        if len(self.memory) < self.train_start:
            return
            
        minibatch = random.sample(self.memory, batch_size)
        states = np.zeros((batch_size, self.state_size[0], self.state_size[1], 1))
        next_states = np.zeros((batch_size, self.state_size[0], self.state_size[1], 1))
        actions, rewards, dones = [], [], []

        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            states[i] = np.reshape(state, [self.state_size[0], self.state_size[1], 1])
            next_states[i] = np.reshape(next_state, [self.state_size[0], self.state_size[1], 1])
            actions.append(action)
            rewards.append(reward)
            dones.append(done)

        # 배치로 한 번에 예측
        target = self.model.predict(states, verbose=0)
        target_next = self.target_model.predict(next_states, verbose=0)

        for i in range(batch_size):
            if dones[i]:
                target[i][actions[i]] = rewards[i]
            else:
                target[i][actions[i]] = rewards[i] + self.gamma * np.amax(target_next[i])

        self.model.fit(states, target, batch_size=batch_size, epochs=1, verbose=0)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
    def save_model(self, episode):
        """모델 저장"""
        try:
            model_path = os.path.join(
                self.config.MODEL_DIR,
                f"tetris_model_ep{episode}.keras"  # .h5 대신 .keras 사용
            )
            self.model.save(model_path)
            
            # 학습 상태 저장
            state_path = os.path.join(
                self.config.MODEL_DIR,
                f"model_state_ep{episode}.json"
            )
            state_data = {
                'epsilon': float(self.epsilon),  # numpy float을 Python float으로 변환
                'episode': int(episode)  # 정수형으로 저장
            }
            with open(state_path, 'w') as f:
                json.dump(state_data, f)
                
            print(f"모델 저장 완료: {model_path}")
        except Exception as e:
            print(f"모델 저장 중 오류 발생: {e}")
            
    def load_model(self):
        """최신 모델 로드"""
        try:
            model_files = glob.glob(os.path.join(self.config.MODEL_DIR, "tetris_model_*.keras"))
            if not model_files:
                print("저장된 모델이 없습니다.")
                return False
                
            latest_model = max(model_files, key=os.path.getctime)
            state_file = latest_model.replace('tetris_model', 'model_state')
            state_file = state_file.replace('.keras', '.json')
            
            self.model = tf.keras.models.load_model(latest_model)
            self.target_model = tf.keras.models.load_model(latest_model)
            
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
        """저장된 메모리 로드"""
        try:
            loaded_memory = self.memory_manager.load_latest_memory()
            if loaded_memory:
                self.memory = deque(loaded_memory, maxlen=self.config.MEMORY_SIZE)
                print(f"메모리 로드 완료: {len(self.memory)} 개의 경험")
        except Exception as e:
            print(f"메모리 로드 실패: {e}")
            
    def save_memory(self, episode):
        """현재 메모리 저장"""
        try:
            self.memory_manager.save_memory(self.memory, episode)
        except Exception as e:
            print(f"메모리 저장 실패: {e}")

# GUI 클래스
class TetrisGUI:
    def __init__(self, config):
        pygame.init()
        self.config = config
        self.block_size = config.BLOCK_SIZE
        self.width = config.BOARD_WIDTH * self.block_size
        self.height = config.BOARD_HEIGHT * self.block_size
        
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Tetris AI')
        
        self.clock = pygame.time.Clock()
        
    def draw(self, state):
        """게임 상태 그리기"""
        self.screen.fill(self.config.COLORS['BLACK'])
        
        for y in range(len(state)):
            for x in range(len(state[0])):
                if state[y][x]:
                    pygame.draw.rect(
                        self.screen,
                        self.config.COLORS['CYAN'],
                        [x * self.block_size, 
                         y * self.block_size,
                         self.block_size - 1, 
                         self.block_size - 1]
                    )
        
        pygame.display.flip()
        self.clock.tick(self.config.FPS)

class TrainingManager:
    def __init__(self, config):
        self.config = config
        self.best_score = float('-inf')
        self.scores_history = []
        self.lines_history = []
        
    def log_episode(self, episode, score, lines_cleared, steps):
        """에피소드 결과 로깅"""
        self.scores_history.append(score)
        self.lines_history.append(lines_cleared)
        
        # 이동 평균 계산
        avg_score = np.mean(self.scores_history[-100:])
        avg_lines = np.mean(self.lines_history[-100:])
        
        # 결과 출력
        print(f"에피소드: {episode}")
        print(f"점수: {score:.2f} (평균: {avg_score:.2f})")
        print(f"제거한 줄: {lines_cleared} (평균: {avg_lines:.2f})")
        print(f"진행 스텝: {steps}")
        print("-" * 50)
        
        # 최고 점수 갱신 체크
        if score > self.best_score:
            self.best_score = score
            return True
        return False

def main():
    # 설정 로드
    config = Config()
    config.load()
    
    # 환경 및 에이전트 초기화
    env = TetrisEnv(config)
    state_size = (config.BOARD_HEIGHT, config.BOARD_WIDTH)
    action_size = env.action_space.n
    
    agent = DQNAgent(state_size, action_size, config)
    state_processor = StateProcessor(config)
    reward_system = RewardSystem(config)
    gui = TetrisGUI(config)
    training_manager = TrainingManager(config)
    
    # 이전 모델 로드 시도
    agent.load_model()
    
    episodes = 1000
    try:
        for episode in range(episodes):
            state = env.reset()
            total_reward = 0
            steps = 0
            lines_cleared = 0
            
            while steps < config.MAX_STEPS_PER_EPISODE:
                # GUI 업데이트
                gui.draw(state)
                
                # 행동 선택 및 실행
                action = agent.act(state)
                next_state, reward, done, _ = env.step(action)
                
                # 상태 처리 및 보상 계산
                processed_state, features = state_processor.process_state(next_state)
                features['game_over'] = done
                reward = reward_system.calculate_reward(features, 
                                                      features['completed_lines'])
                
                # 경험 저장
                agent.remember(state, action, reward, next_state, done)
                
                # 학습 수행
                if len(agent.memory) >= agent.batch_size:
                    agent.replay(agent.batch_size)
                
                # 상태 업데이트
                state = next_state
                total_reward += reward
                steps += 1
                lines_cleared += features['completed_lines']
                
                # 주기적 메모리 정리
                if steps % 100 == 0:
                    gc.collect()
                
                # 게임 종료 체크
                if done:
                    break
                
                # 이벤트 처리
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        # 종료 시 저장
                        agent.save_model(episode)
                        agent.save_memory(episode)
                        pygame.quit()
                        return
            
            # 에피소드 결과 로깅
            is_best = training_manager.log_episode(episode, total_reward, 
                                                 lines_cleared, steps)
            
            # 주기적 저장
            if (episode + 1) % config.SAVE_INTERVAL == 0 or is_best:
                agent.save_model(episode)
                agent.save_memory(episode)
            
            # 타겟 네트워크 업데이트
            if episode % config.TARGET_UPDATE_FREQUENCY == 0:
                agent.update_target_model()
                
    except KeyboardInterrupt:
        print("\n학습 중단...")
    except Exception as e:
        print(f"오류 발생: {e}")
    finally:
        # 최종 상태 저장
        agent.save_model(episode)
        agent.save_memory(episode)
        pygame.quit()
        
    # 학습 결과 저장
    config.save()

if __name__ == "__main__":
    # 시작 전 메모리 정리
    gc.enable()
    tf.keras.backend.clear_session()
    
    try:
        main()
    except Exception as e:
        print(f"프로그램 실행 중 오류 발생: {e}")
        traceback.print_exc()
    finally:
        # 프로그램 종료 시 정리
        pygame.quit()
        gc.collect()