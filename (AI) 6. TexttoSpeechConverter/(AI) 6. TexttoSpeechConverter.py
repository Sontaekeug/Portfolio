import os
import threading
import tkinter as tk
from tkinter import ttk, messagebox    # tkinter: Python 내장 GUI 라이브러리 (창, 버튼, 메시지 박스 등 UI 요소 제공)
import speech_recognition as sr        # speech_recognition: 음성을 텍스트로 변환(STT)하는 라이브러리
from gtts import gTTS                  # gTTS: 텍스트를 음성으로 변환(TTS)하는 라이브러리
from pydub import AudioSegment
from pydub.playback import play        # pydub: 오디오 파일 변환 및 재생을 지원하는 라이브러리
import tempfile                        # tempfile: 임시 파일 생성 및 관리
from PIL import Image, ImageTk         # PIL: 이미지를 tkinter에서 사용하기 위한 라이브러리

class VoiceConverter:
    """음성-텍스트 변환기 GUI 클래스"""
    
    def __init__(self, root):
        """GUI 초기화"""
        self.root = root
        self.root.title("음성-텍스트 변환기")
        self.root.geometry("600x400")
        
        # 테마 색상 설정 (Kakao 스타일)
        self.colors = {
            'primary': '#FFE812',
            'secondary': '#3C1E1E',
            'background': '#FFFFFF',
            'text': '#3C1E1E',
            'button_text': '#3C1E1E',
            'chat_bg': '#F8F8F8'
        }
        
        # tkinter 스타일 설정
        self.style = ttk.Style()
        self.style.theme_use('clam')  # 테마 설정
        self.style.configure('Kakao.TButton', padding=10, background=self.colors['primary'],
                             foreground=self.colors['button_text'], font=('나눔고딕', 10, 'bold'),
                             borderwidth=0, relief='flat')  # 버튼 스타일
        self.style.configure('Kakao.TFrame', background=self.colors['background'])  # 프레임 스타일
        
        # 메인 프레임 생성
        self.main_frame = ttk.Frame(root, style='Kakao.TFrame')
        self.main_frame.pack(expand=True, fill='both', padx=20, pady=20)  # 창 크기 조정

        # 탭 컨트롤 생성 (STT, TTS 기능 구현)
        self.tab_control = ttk.Notebook(self.main_frame)
        self.setup_stt_tab()
        self.setup_tts_tab()
        self.tab_control.pack(expand=True, fill='both')

        # 음성 인식기 초기화 (STT에 사용)
        self.recognizer = sr.Recognizer()

        # 상태 표시줄 생성
        self.status_var = tk.StringVar()  # 상태 메시지 저장 변수
        self.status_bar = ttk.Label(self.root, textvariable=self.status_var,
                                    font=('나눔고딕', 9), background=self.colors['secondary'],
                                    foreground=self.colors['background'], padding=5)
        self.status_bar.pack(side='bottom', fill='x')  # 상태 표시줄 배치

    def setup_stt_tab(self):
        """
        STT (음성 → 텍스트) 탭 구성
        - 음성을 텍스트로 변환하는 기능 제공
        """
        self.stt_tab = ttk.Frame(self.tab_control, style='Kakao.TFrame')  # 탭 생성
        self.tab_control.add(self.stt_tab, text='음성→텍스트')  # STT 탭 추가
        
        # 음성 녹음 버튼
        self.record_button = ttk.Button(self.stt_tab, text="음성 녹음 시작", style='Kakao.TButton',
                                        command=self.toggle_recording)
        self.record_button.pack(pady=15)

        # 음성 인식 결과를 표시할 텍스트 영역
        self.text_output = tk.Text(self.stt_tab, height=10, width=50, font=('나눔고딕', 10),
                                   bg=self.colors['chat_bg'], fg=self.colors['text'], relief='flat',
                                   padx=15, pady=10, spacing1=5, spacing2=5)
        self.text_output.pack(pady=15, padx=20)

        # 텍스트 저장 버튼
        self.save_button = ttk.Button(self.stt_tab, text="텍스트 저장", style='Kakao.TButton',
                                      command=self.save_text)
        self.save_button.pack(pady=10)

        self.is_recording = False  # 녹음 상태 변수 초기화

    def setup_tts_tab(self):
        """
        TTS (텍스트 → 음성) 탭 구성
        - 텍스트를 음성으로 변환하는 기능 제공
        """
        self.tts_tab = ttk.Frame(self.tab_control, style='Kakao.TFrame')  # 탭 생성
        self.tab_control.add(self.tts_tab, text='텍스트→음성')  # TTS 탭 추가
        
        # 텍스트 입력 영역
        self.text_input = tk.Text(self.tts_tab, height=10, width=50, font=('나눔고딕', 10),
                                  bg=self.colors['chat_bg'], fg=self.colors['text'], relief='flat',
                                  padx=15, pady=10, spacing1=5, spacing2=5)
        self.text_input.pack(pady=15, padx=20)

        # 버튼 프레임 생성 (텍스트 변환 및 재생 버튼 배치)
        button_frame = ttk.Frame(self.tts_tab, style='Kakao.TFrame')
        button_frame.pack(pady=10)
        
        # 텍스트 → 음성 변환 버튼
        self.convert_button = ttk.Button(button_frame, text="음성으로 변환", style='Kakao.TButton',
                                         command=self.convert_to_speech)
        self.convert_button.pack(side='left', padx=5)
        
        # 음성 재생 버튼
        self.play_button = ttk.Button(button_frame, text="음성 재생", style='Kakao.TButton',
                                      command=self.play_speech)
        self.play_button.pack(side='left', padx=5)

    def toggle_recording(self):
        """녹음 시작/중지 토글"""
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self):
        """녹음 시작"""
        self.is_recording = True
        self.record_button.configure(text="녹음 중지")
        self.status_var.set("녹음 중...")

        # 녹음을 백그라운드에서 처리하기 위해 스레드 사용
        self.record_thread = threading.Thread(target=self.record_audio)
        self.record_thread.start()

    def stop_recording(self):
        """녹음 중지"""
        self.is_recording = False
        self.record_button.configure(text="음성 녹음 시작")
        self.status_var.set("녹음 완료")

    def record_audio(self):
        """음성을 텍스트로 변환 (Google Speech-to-Text API 사용)"""
        with sr.Microphone() as source:
            self.recognizer.adjust_for_ambient_noise(source)  # 주변 소음 조정
            try:
                while self.is_recording:
                    audio = self.recognizer.listen(source, timeout=5)
                    try:
                        text = self.recognizer.recognize_google(audio, language='ko-KR')  # 한국어 인식
                        self.text_output.insert('end', text + '\n')  # 인식된 텍스트 출력
                    except sr.UnknownValueError:
                        self.status_var.set("음성을 인식할 수 없습니다.")
                    except sr.RequestError:
                        self.status_var.set("Google Speech API 오류 발생")
            except Exception as e:
                self.status_var.set(f"오류 발생: {str(e)}")

    def save_text(self):
        """텍스트 저장"""
        text = self.text_output.get('1.0', 'end-1c')  # 텍스트 가져오기
        if text:
            try:
                with open('transcribed_text.txt', 'w', encoding='utf-8') as file:
                    file.write(text)
                messagebox.showinfo("저장 완료", "텍스트가 저장되었습니다.")
            except Exception as e:
                messagebox.showerror("저장 실패", f"저장 중 오류 발생: {str(e)}")
        else:
            messagebox.showwarning("경고", "저장할 텍스트가 없습니다.")

    def convert_to_speech(self):
        """텍스트를 음성으로 변환"""
        text = self.text_input.get('1.0', 'end-1c')  # 입력 텍스트 가져오기
        if text:
            try:
                self.status_var.set("음성 변환 중...")
                tts = gTTS(text=text, lang='ko')  # gTTS로 음성 변환
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
                    self.temp_audio_file = fp.name  # 임시 파일 경로 저장
                    tts.save(self.temp_audio_file)
                self.status_var.set("음성 변환 완료")
                messagebox.showinfo("변환 완료", "텍스트가 음성으로 변환되었습니다.")
            except Exception as e:
                self.status_var.set(f"변환 중 오류 발생: {str(e)}")
                messagebox.showerror("변환 실패", f"오류 발생: {str(e)}")
        else:
            messagebox.showwarning("경고", "변환할 텍스트를 입력하세요.")

    def play_speech(self):
        """변환된 음성을 재생"""
        if hasattr(self, 'temp_audio_file') and os.path.exists(self.temp_audio_file):
            try:
                self.status_var.set("음성 재생 중...")
                audio = AudioSegment.from_mp3(self.temp_audio_file)  # 임시 파일에서 음성 로드
                play(audio)  # 음성 재생
                self.status_var.set("재생 완료")
            except Exception as e:
                self.status_var.set(f"재생 중 오류 발생: {str(e)}")
                messagebox.showerror("재생 실패", f"오류 발생: {str(e)}")
        else:
            messagebox.showwarning("경고", "재생할 음성이 없습니다.")
            
# 메인 실행
if __name__ == "__main__":
    root = tk.Tk()
    app = VoiceConverter(root)
    root.mainloop()
