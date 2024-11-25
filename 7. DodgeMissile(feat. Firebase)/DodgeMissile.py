import pygame
import random
import sys
import time
import json
import os
from tkinter import *
from tkinter import simpledialog
import firebase_admin
from firebase_admin import credentials, db, auth
import requests

# Firebase Admin SDK 초기화
# 서비스 계정 키 파일 경로 지정
cred = credentials.Certificate(r"C:\Users\Adle\Desktop\DodgeMissile\dodgemissilerank-firebase-adminsdk-2wtxk-248dafa4e3.json")

# Firebase 프로젝트 URL 지정 (수정)
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://dodgemissilerank-default-rtdb.firebaseio.com'  # URL 수정
})

# 초기화
pygame.init()

# 이미지 로드 (수정된 부분)
current_dir = r"C:\Users\Adle\Desktop\DodgeMissile"
player_img = pygame.image.load(os.path.join(current_dir, 'player.png'))
player_img_left = pygame.image.load(os.path.join(current_dir, 'playerL.png'))
player_img_right = pygame.image.load(os.path.join(current_dir, 'playerR.png'))

# 이미지 크기 조정
player_img = pygame.transform.scale(player_img, (25, 25))
player_img_left = pygame.transform.scale(player_img_left, (25, 25))
player_img_right = pygame.transform.scale(player_img_right, (25, 25))

# 미사일 색상 정의 추가
ORANGE = (255, 140, 0)
LIGHT_ORANGE = (255, 180, 100)

# 랭킹 파일 관리
RANKING_FILE = "ranking.json"
def save_ranking(name, play_time):
    try:
        ref = db.reference('/rankings')  # 경로 수정
        new_ranking = {
            "name": name,
            "time": play_time,
            "timestamp": int(time.time())
        }
        ref.push(new_ranking)
        print("Ranking saved successfully")
    except Exception as e:
        print(f"Error saving ranking: {str(e)}")

def save_ranking_with_auth(name, play_time, id_token):
    ref = db.reference('rankings')
    new_ranking = {
        "name": name,
        "time": play_time,
        "timestamp": int(time.time()),
        "authenticated": True
    }
    try:
        ref.push(new_ranking)
    except Exception as e:
        print(f"Error saving authenticated ranking: {e}")
        save_ranking(name, play_time)

def create_user(email, password):
    user = auth.create_user(
        email=email,
        password=password
    )
    print('Successfully created new user:', user.uid)

def login_user(email, password):
    api_key = 'AIzaSyBtfqepUp9zMcTuPGdebQy8KQ2bvTVed6k'
    url = f'https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={api_key}'
    payload = {
        'email': email,
        'password': password,
        'returnSecureToken': True
    }
    response = requests.post(url, data=payload)
    return response.json()

# 메뉴 색상 정의
MENU_BG = (50, 50, 50)
MENU_TEXT = (255, 255, 255)
MENU_HOVER = (100, 100, 100)

# 폰트 설정 수정
pygame.font.init()
try:
    game_font = pygame.font.SysFont('malgun gothic', 36)  # 기본 폰트
    small_font = pygame.font.SysFont('malgun gothic', 18)  # 작은 폰트 (1/2 크기)
    time_font = pygame.font.SysFont('malgun gothic', 29)  # 시간 폰트 (80% 크기)
except:
    game_font = pygame.font.SysFont('arial', 36)
    small_font = pygame.font.SysFont('arial', 18)
    time_font = pygame.font.SysFont('arial', 29)

# 창 크기 설정 추가
WINDOW_SIZES = [
    (800, 600),
    (1024, 768),
    (1280, 720)
]

class Menu:
    def __init__(self, screen, font):
        self.screen = screen
        self.font = game_font  # 기존 font를 새로운 폰트로 변경
        self.options = ['게임 시작', '랭킹 보기', '종료']
        self.selected = None
        self.current_option = 0  # 현재 선택된 메뉴 항목
        
    def draw(self):
        self.screen.fill((0, 0, 0))
        menu_height = 80
        for i, option in enumerate(self.options):
            text = self.font.render(option, True, MENU_TEXT)
            rect = text.get_rect(center=(400, 200 + i * menu_height))
            
            # 키보드 선택 표시
            if i == self.current_option:
                pygame.draw.rect(self.screen, MENU_HOVER, rect.inflate(20, 20))
            
            self.screen.blit(text, rect)
        pygame.display.flip()

    def handle_click(self):
        return self.selected if self.selected is not None else -1

    def handle_input(self, event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                self.current_option = (self.current_option - 1) % len(self.options)
            elif event.key == pygame.K_DOWN:
                self.current_option = (self.current_option + 1) % len(self.options)
            elif event.key == pygame.K_SPACE:
                return self.current_option
        return -1

class Game:
    def __init__(self):
        self.current_size_index = 0
        self.WIDTH, self.HEIGHT = WINDOW_SIZES[self.current_size_index]
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT), pygame.RESIZABLE)
        
        # 게임 변수
        self.player_size = 50
        self.player = pygame.Rect(self.WIDTH//2, self.HEIGHT-2*self.player_size, 
                                self.player_size, self.player_size)
        self.missile_list = []
        self.start_time = None  # 게임 시작 시간을 None으로 초기화
        self.clock = pygame.time.Clock()
        self.font = game_font  # 기존 font 설정을 새로운 폰트로 변경
        self.small_font = small_font
        self.time_font = time_font
        self.missile_radius = 2  # 미사일 크기
        self.player_hitbox = pygame.Rect(0, 0, 12.5, 12.5)  # 충돌 판정용 히트박스
        self.base_missile_speed = 2  # 기본 미사일 속도
        self.missile_speed_increment = 0  # 미사일 속도 증가분
        self.missile_per_wave = 10  # 초기 미사일 수를 10개
        self.last_spawn_time = None  # 마지막 미사일 생성 시간도 None으로 초기화
        self.menu = Menu(self.screen, self.font)
        self.state = 'menu'  # 'menu', 'game', 'ranking' 상태 추가
        self.reset_game()  # 새로운 메서드 추가
        self.missile_update_time = 2  # 3초마다 미사일 수 증가
        self.last_missile_increase = 0  # 마지막 미사일 증가 시간
        self.play_time = 0
        self.current_player_img = player_img  # 현재 플레이어 이미지 추적
        self.paused = False  # 게임 일시정지 상태 추가
        self.player_hitbox = pygame.Rect(0, 0, 12.5, 12.5)  # 초기 히트박스 크기 설정 (50% 크기)
    
    def reset_game(self):
        """게임 상태를 초기화하는 메서드"""
        self.missile_list = []
        self.start_time = None
        self.last_spawn_time = None
        self.missile_speed_increment = 0
        self.missile_per_wave = 5
        self.player.center = (self.WIDTH//2, self.HEIGHT-2*self.player_size)
        self.paused = False  # 게임 초기화 시 일시정지 해제
        self.last_missile_increase = 0  # 마지막 미사일 증가 시간 초기화
        
    def update_player_hitbox(self):
        # 플레이어 이미지의 몸체에 맞게 히트박스 위치 및 크기 조정
        self.player_hitbox.size = (self.player.width * 0.5, self.player.height * 0.5)  # 히트박스 크기 설정
        self.player_hitbox.center = (                                # 히트박스 센터 설정
        self.player.center[0] - 0.25 * self.player.width,
        self.player.center[1] - 0.25 * self.player.height
    )
        
    def create_missile(self, current_time):
        side = random.randint(0, 3)  # 0:위, 1:오른쪽, 2:아래, 3:왼쪽
        if side == 0:  # 위
            x = random.randint(0, self.WIDTH)
            y = -20
        elif side == 1:  # 오른쪽
            x = self.WIDTH + 20
            y = random.randint(0, self.HEIGHT)
        elif side == 2:  # 아래
            x = random.randint(0, self.WIDTH)
            y = self.HEIGHT + 20
        else:  # 왼쪽
            x = -20
            y = random.randint(0, self.HEIGHT)
            
        # 생성 시점의 플레이어 위치 저장
        target_x, target_y = self.player.center
        
        # 방향 벡�� 계산
        dx = target_x - x
        dy = target_y - y
        
        # 정규화
        length = (dx**2 + dy**2)**0.5
        if length > 0:
            dx = dx/length
            dy = dy/length
            
        return {
            'rect': pygame.Rect(x, y, 4, 4).inflate(-1.2, -1.2),  # 미사일 히트박스를 70% 크기로 줄임
            'dx': dx,
            'dy': dy,
            'speed': self.base_missile_speed + self.missile_speed_increment
        }

    def update_missile(self, missile):
        # 저장된 방향으로 이동
        missile['rect'].x += missile['dx'] * missile['speed']
        missile['rect'].y += missile['dy'] * missile['speed']
        
    def draw_missile(self, missile):
        x, y = missile['rect'].topleft
        pygame.draw.circle(self.screen, ORANGE, (x + 2, y + 2), self.missile_radius)
        pygame.draw.circle(self.screen, LIGHT_ORANGE, (x + 1, y + 1), self.missile_radius - 1)
    
    def show_ranking(self):
        try:
            ref = db.reference('/rankings')  # 경로 수정
            rankings = ref.get()
            
            if (rankings):
                # 딕셔너리를 리스트로 변환
                ranking_list = []
                for key, value in rankings.items():
                    ranking_list.append(value)
                
                # 시간 내림차순으로 정렬
                sorted_rankings = sorted(ranking_list, key=lambda x: x['time'], reverse=True)
                
                self.screen.fill((0, 0, 0))
                title = self.font.render("Top 5 Rankings", True, MENU_TEXT)
                self.screen.blit(title, (300, 50))
                
                for i, rank in enumerate(sorted_rankings[:5], 1):
                    text = self.font.render(
                        f"{i}. {rank['name']}: {rank['time']:.1f}초",
                        True, MENU_TEXT
                    )
                    self.screen.blit(text, (200, 100 + i * 50))
                
                back = self.font.render("Press ESC to return", True, MENU_TEXT)
                self.screen.blit(back, (300, 500))
                pygame.display.flip()

                waiting = True
                while waiting:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            pygame.quit()
                            sys.exit()
                        if event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_ESCAPE:
                                waiting = False
                                self.state = 'menu'
            else:
                # 랭킹이 없을 때 표시
                self.screen.fill((0, 0, 0))
                no_ranking_text = self.font.render("No rankings available", True, MENU_TEXT)
                self.screen.blit(no_ranking_text, (300, 250))
                pygame.display.flip()
                
        except Exception as e:
            print(f"Error loading rankings: {str(e)}")
            # 에러 메시지 화면에 표시
            self.screen.fill((0, 0, 0))
            error_text = self.font.render("Error loading rankings", True, MENU_TEXT)
            self.screen.blit(error_text, (300, 250))
            pygame.display.flip()

    def game_over(self):
        root = Tk()
        root.withdraw()
        
        self.play_time = time.time() - self.start_time
        
        name = simpledialog.askstring("Game Over", "이름을 입력하세요:", parent=root)
        
        if name:
            try:
                ref = db.reference('rankings')
                new_ranking = {
                    "name": name,
                    "time": self.play_time,
                    "timestamp": int(time.time())
                }
                ref.push(new_ranking)
            except Exception as e:
                print(f"Error saving ranking: {e}")
        
        root.destroy()
        self.state = 'menu'

    def toggle_window_size(self):
        self.current_size_index = (self.current_size_index + 1) % len(WINDOW_SIZES)
        self.WIDTH, self.HEIGHT = WINDOW_SIZES[self.current_size_index]
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT), pygame.RESIZABLE)
        # 플레이어 위치 재조정
        self.player.center = (self.WIDTH//2, self.HEIGHT-2*self.player_size)

    def run(self):
        running = True
        while running:
            if self.state == 'menu':
                self.menu.draw()
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_F11:  # F11로 창 크기 변경
                            self.toggle_window_size()
                        else:
                            clicked = self.menu.handle_input(event)
                            if clicked >= 0:
                                if clicked == 0:  # 게임 시작
                                    self.state = 'game'
                                    self.reset_game()
                                    self.start_game()
                                elif clicked == 1:  # 랭킹 보기
                                    self.state = 'ranking'
                                    self.show_ranking()
                                elif clicked == 2:  # 종료
                                    running = False
            
            elif self.state == 'game':
                self.start_game()
            
            self.clock.tick(60)

    def start_game(self):
        self.start_time = time.time()
        self.last_spawn_time = self.start_time
        last_second = 0

        for _ in range(self.missile_per_wave):
            self.missile_list.append(self.create_missile(self.start_time))

        while self.state == 'game':
            if self.paused:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.state = 'menu'
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            self.paused = False  # ESC 키로 일시정지 해제
                continue

            current_time = time.time() - self.start_time
            current_second = int(current_time)
            
            # 3초마다 미사일 수 증가로 수정
            if current_time - self.last_missile_increase >= self.missile_update_time:
                self.missile_per_wave += 1
                self.last_missile_increase = current_time
                
                # 미사일 생성
                for _ in range(self.missile_per_wave):
                    self.missile_list.append(self.create_missile(current_time))

            # 이벤트 처리
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.state = 'menu'
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_F11:
                        self.toggle_window_size()
                    elif event.key == pygame.K_ESCAPE:
                        self.state = 'menu'  # ESC 키로 메인 화면으로 나가기

            # 플레이어 이동 및 이미지 업데이트 (4방향)
            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT] and self.player.left > 0:
                self.player.x -= 5
                self.current_player_img = player_img_left
            elif keys[pygame.K_RIGHT] and self.player.right < self.WIDTH:
                self.player.x += 5
                self.current_player_img = player_img_right
            else:
                self.current_player_img = player_img
                
            if keys[pygame.K_UP] and self.player.top > 0:
                self.player.y -= 5
            if keys[pygame.K_DOWN] and self.player.bottom < self.HEIGHT:
                self.player.y += 5
            
            self.update_player_hitbox()

            # 미사일 이동 및 충돌 검사
            for missile in self.missile_list[:]:
                self.update_missile(missile)
                # 충돌 검사만 수행
                if self.player_hitbox.colliderect(missile['rect']):
                    self.game_over()
                    return

            # 화면 그리기
            self.screen.fill((0, 0, 0))
            self.screen.blit(self.current_player_img, self.player)
            # pygame.draw.rect(self.screen, (255, 255, 255), self.player_hitbox, 1)  # 히트박스 그리기
            for missile in self.missile_list:
                self.draw_missile(missile)

            # 시간만 표시 (크기 조정)
            time_text = self.time_font.render(f'Time: {current_time:.1f}초', True, (255, 255, 255))
            self.screen.blit(time_text, (10, 10))

            pygame.display.flip()
            self.clock.tick(60)

# 게임 시작 시 사용자 로그인
def start_game():
    email = input("Enter your email: ")
    password = input("Enter your password: ")
    login_response = login_user(email, password)
    if 'idToken' in login_response:
        id_token = login_response['idToken']
        print("Login successful!")
        # 게임 로직 실행
        run_game(id_token)
    else:
        print("Login failed!")

# 게임 로직
def run_game(id_token):
    name = "Player1"  # 예시 이름
    play_time = 120  # 예시 플레이 시간
    save_ranking_with_auth(name, play_time, id_token)

if __name__ == "__main__":
    game = Game()
    game.run()
    pygame.quit()