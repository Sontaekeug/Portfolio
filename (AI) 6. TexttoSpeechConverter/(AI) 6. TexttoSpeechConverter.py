import tkinter as tk
from tkinter import ttk, messagebox
import speech_recognition as sr
from gtts import gTTS
import os
import threading
from pydub import AudioSegment
from pydub.playback import play
import tempfile
from PIL import Image, ImageTk

class VoiceConverter:
    def __init__(self, root):
        self.root = root
        self.root.title("음성-텍스트 변환기")
        self.root.geometry("600x400")
        
        # 테마 색상
        self.colors = {
            'primary': '#FFE812',        # 노란색
            'secondary': '#3C1E1E',      # 갈색
            'background': '#FFFFFF',      # 흰색 배경
            'text': '#3C1E1E',           # 텍스트 색상
            'button_text': '#3C1E1E',    # 버튼 텍스트 색상
            'chat_bg': '#F8F8F8'         # 채팅창 배경색
        }
        
        # 스타일 설정
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # 버튼 스타일
        self.style.configure('Kakao.TButton',
                           padding=10,
                           background=self.colors['primary'],
                           foreground=self.colors['button_text'],
                           font=('나눔고딕', 10, 'bold'),
                           borderwidth=0,
                           relief='flat')
        
        # 탭 스타일
        self.style.configure('Kakao.TNotebook',
                           background=self.colors['background'])
        self.style.configure('Kakao.TNotebook.Tab',
                           padding=[15, 8],
                           font=('나눔고딕', 10, 'bold'),
                           background=self.colors['background'],
                           foreground=self.colors['secondary'])
        
        # 프레임 스타일
        self.style.configure('Kakao.TFrame',
                           background=self.colors['background'])
        
        # 메인 프레임 설정
        self.root.configure(bg=self.colors['background'])
        self.main_frame = ttk.Frame(root, style='Kakao.TFrame')
        self.main_frame.pack(expand=True, fill='both', padx=20, pady=20)
        
        # 탭 컨트롤
        self.tab_control = ttk.Notebook(self.main_frame, style='Kakao.TNotebook')
        
        # 이미지 로드 및 크기 조정
        mic_img = Image.open("mic_icon.png")  # 마이크 이미지 파일
        speaker_img = Image.open("speaker_icon.png")  # 스피커 이미지 파일
        
        # 이미지 크기 조정 (예: 20x20 픽셀)
        mic_img = mic_img.resize((20, 20), Image.Resampling.LANCZOS)
        speaker_img = speaker_img.resize((20, 20), Image.Resampling.LANCZOS)
        
        # PhotoImage 객체 생성
        self.mic_icon = ImageTk.PhotoImage(mic_img)
        self.speaker_icon = ImageTk.PhotoImage(speaker_img)
        
        # 탭 레이블 생성
        self.stt_label = ttk.Label(self.tab_control, 
                                 text="음성→텍스트  ",  # 공백으로 이미지와 간격 조정
                                 image=self.mic_icon,
                                 compound='right',
                                 background=self.colors['background'])
        
        self.tts_label = ttk.Label(self.tab_control,
                                 text="텍스트→음성  ",  # 공백으로 이미지와 간격 조정
                                 image=self.speaker_icon,
                                 compound='right',
                                 background=self.colors['background'])
        
        # STT 탭
        self.stt_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(self.stt_tab, text='', image=self.mic_icon)
        
        # TTS 탭
        self.tts_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(self.tts_tab, text='', image=self.speaker_icon)
        
        # 탭 레이블 설정
        self.tab_control.tab(0, text='음성→텍스트')
        self.tab_control.tab(1, text='텍스트→음성')
        
        self.tab_control.pack(expand=True, fill='both')
        
        self.setup_stt_tab()
        self.setup_tts_tab()
        
        # 음성 인식기 초기화
        self.recognizer = sr.Recognizer()
        
        # 카카오톡 스타일 상태 표시줄
        self.status_var = tk.StringVar()
        self.status_bar = ttk.Label(self.root,
                                  textvariable=self.status_var,
                                  font=('나눔고딕', 9),
                                  background=self.colors['secondary'],
                                  foreground=self.colors['background'],
                                  padding=5)
        self.status_bar.pack(side='bottom', fill='x')
        
    def setup_stt_tab(self):
        # 녹음 버튼
        self.record_button = ttk.Button(self.stt_tab, text="음성 녹음 시작",
                                      style='Kakao.TButton',
                                      command=self.toggle_recording)
        self.record_button.pack(pady=15)
        
        # 채팅창 스타일의 텍스트 출력 영역
        self.text_output = tk.Text(self.stt_tab, height=10, width=50,
                                 font=('나눔고딕', 10),
                                 bg=self.colors['chat_bg'],
                                 fg=self.colors['text'],
                                 relief='flat',
                                 padx=15,
                                 pady=10,
                                 spacing1=5,
                                 spacing2=5)
        self.text_output.pack(pady=15, padx=20)
        
        # 저장 버튼
        self.save_button = ttk.Button(self.stt_tab, text="텍스트 저장",
                                    style='Kakao.TButton',
                                    command=self.save_text)
        self.save_button.pack(pady=10)
        
        self.is_recording = False
        
    def setup_tts_tab(self):
        # 채팅창 스타일의 텍스트 입력 영역
        self.text_input = tk.Text(self.tts_tab, height=10, width=50,
                                font=('나눔고딕', 10),
                                bg=self.colors['chat_bg'],
                                fg=self.colors['text'],
                                relief='flat',
                                padx=15,
                                pady=10,
                                spacing1=5,
                                spacing2=5)
        self.text_input.pack(pady=15, padx=20)
        
        # 버튼 프레임
        button_frame = ttk.Frame(self.tts_tab, style='Kakao.TFrame')
        button_frame.pack(pady=10)
        
        # 음성 변환 버튼
        self.convert_button = ttk.Button(button_frame, text="음성으로 변환",
                                       style='Kakao.TButton',
                                       command=self.convert_to_speech)
        self.convert_button.pack(side='left', padx=5)
        
        # 음성 재생 버튼
        self.play_button = ttk.Button(button_frame, text="음성 재생",
                                    style='Kakao.TButton',
                                    command=self.play_speech)
        self.play_button.pack(side='left', padx=5)
        
    def toggle_recording(self):
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()
            
    def start_recording(self):
        self.is_recording = True
        self.record_button.configure(text="녹음 중지")
        self.status_var.set("녹음 중...")
        
        # 녹음 스레드 시작
        self.record_thread = threading.Thread(target=self.record_audio)
        self.record_thread.start()
        
    def stop_recording(self):
        self.is_recording = False
        self.record_button.configure(text="음성 녹음 시작")
        self.status_var.set("녹음 완료")
        
    def record_audio(self):
        with sr.Microphone() as source:
            self.recognizer.adjust_for_ambient_noise(source)
            try:
                while self.is_recording:
                    audio = self.recognizer.listen(source, timeout=5)
                    try:
                        text = self.recognizer.recognize_google(audio, language='ko-KR')
                        self.text_output.insert('end', text + '\n')
                    except sr.UnknownValueError:
                        self.status_var.set("음성을 인식할 수 없습니다.")
                    except sr.RequestError:
                        self.status_var.set("Google Speech API 오류가 발생했습니다.")
            except Exception as e:
                self.status_var.set(f"오류 발생: {str(e)}")
                
    def save_text(self):
        text = self.text_output.get('1.0', 'end-1c')
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
        text = self.text_input.get('1.0', 'end-1c')
        if text:
            try:
                self.status_var.set("음성 변환 중...")
                tts = gTTS(text=text, lang='ko')
                
                # 임시 파일로 저장
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
                    self.temp_audio_file = fp.name
                    tts.save(self.temp_audio_file)
                
                self.status_var.set("음성 변환 완료")
                messagebox.showinfo("변환 완료", "텍스트가 음성으로 변환되었습니다.")
            except Exception as e:
                self.status_var.set(f"변환 중 오류 발생: {str(e)}")
                messagebox.showerror("변환 실패", f"음성 변환 중 오류 발생: {str(e)}")
        else:
            messagebox.showwarning("경고", "변환할 텍스트를 입력하세요.")
            
    def play_speech(self):
        if hasattr(self, 'temp_audio_file') and os.path.exists(self.temp_audio_file):
            try:
                self.status_var.set("음성 재생 중...")
                # ffmpeg 경로 명시적 설정
                AudioSegment.converter = r"C:\Program Files\ffmpeg\bin\ffmpeg.exe"  # ffmpeg 설치 경로에 맞게 수정
                audio = AudioSegment.from_mp3(self.temp_audio_file)
                play(audio)
                self.status_var.set("재생 완료")
            except FileNotFoundError:
                messagebox.showerror("오류", "FFmpeg가 설치되지 않았습니다. FFmpeg를 설치해주세요.")
            except Exception as e:
                self.status_var.set(f"재생 중 오류 발생: {str(e)}")
                messagebox.showerror("재생 실패", f"음성 재생 중 오류 발생: {str(e)}")
        else:
            messagebox.showwarning("경고", "재생할 음성 파일이 없습니다.")

if __name__ == "__main__":
    root = tk.Tk()
    app = VoiceConverter(root)
    root.mainloop()