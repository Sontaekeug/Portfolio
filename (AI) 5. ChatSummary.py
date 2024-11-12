import sys
import re
from collections import defaultdict, Counter
from typing import List, Dict, Tuple
from datetime import datetime

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QPushButton, QTextEdit, QLabel, 
                           QFileDialog, QProgressBar, QMessageBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QPalette, QColor

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from konlpy.tag import Okt

class ChatThemeAnalyzer:
    def __init__(self):
        # 한국어 형태소 분석기 초기화
        self.okt = Okt()
        
        # 한국어 기본 불용어 정의
        self.stop_words = {
            '이', '그', '저', '것', '수', '네', '예', '응', '음',
            '아', '어', '있', '없', '하', '되', '을', '를', '이', 
            '가', '은', '는', '와', '과', '도', '에', '에서', '으로', 
            '로', '에게', '뭐', '왜', '어떻게'
        }
        
        # 욕설 필터링 단어 목록
        self.curse_words = {
            '시발', '씨발', '시팔', '씨팔', '죽여', '새끼', '시바', '씨바',
            '닥쳐', '꺼져', '미친', '병신', 'ㅅㅂ', 'ㅆㅂ', 'ㅄ', 'ㅂㅅ'
        }
        
        # 무시할 감탄사/이모티콘 목록
        self.ignore_expressions = {
            'ㅋ', 'ㅎ', 'ㅠ', 'ㅜ', 'ㅡ', 'ㅗ', 'ㅛ', 'ㄷ', 'ㄱ',
            'ㅇㅇ', 'ㄴㄴ', 'ㅇㅋ', 'ㅊㅊ', '아니', '어', '음', '아',
            '오', '엥', '헐', '와', '워', '웅', '윽', '잉', '네', '넹',
            '응', '어휴', '아휴', '흠', '힝', '?', '??', '???',
            '!', '!!', '!!!', '.', '..', '...', '_'
        }
    
    def parse_chat(self, chat_text: str) -> List[Dict]:
        """채팅 텍스트를 파싱하여 메시지 리스트로 변환"""
        messages = []
        current_conversation = []
        
        # 정규표현식 패턴
        pattern = r'\[(.*?)\] \[(.+?)\] (.*)'
        
        for line in chat_text.split('\n'):
            line = line.strip()
            if line:
                match = re.match(pattern, line)
                if match:
                    username, time, content = match.groups()
                    messages.append({
                        'username': username,
                        'time': time,
                        'content': content.strip(),
                    })
        
        return messages

    def clean_text(self, text: str) -> str:
        """텍스트 정제"""
        # 욕설 필터링
        cleaned_text = text
        for curse in self.curse_words:
            cleaned_text = cleaned_text.replace(curse, '***')
        
        # 단어 단위로 분리
        words = cleaned_text.split()
        
        # 감탄사, 이모티콘 제거
        words = [word for word in words if word not in self.ignore_expressions]
        
        # 연속된 특수문자 제거
        words = [word for word in words if not all(c in '!.?ㅋㅎㅠㅜㅡ' for c in word)]
        
        # 의미 있는 단어만 남기기
        words = [word for word in words if len(word.strip()) > 1]
        
        return ' '.join(words)

    def preprocess_text(self, text: str) -> str:
        """텍스트 전처리"""
        # 기본 전처리
        text = text.lower()
        
        # 정제
        text = self.clean_text(text)
        
        # 형태소 분석
        morphs = self.okt.normalize(text)  # 정규화
        morphs = self.okt.morphs(morphs)  # 형태소 분석
        
        # 불용어 제거 및 필터링
        words = [word for word in morphs 
                if word not in self.stop_words 
                and word not in self.ignore_expressions
                and len(word) > 1
                and not word.isdigit()]
        
        return ' '.join(words)

    def group_consecutive_messages(self, messages: List[Dict]) -> List[Dict]:
        """연속된 메시지를 하나의 대화 단위로 그룹화"""
        grouped = []
        current_group = []
        current_user = None

        for msg in messages:
            if current_user != msg['username']:
                if current_group:
                    combined_content = ' '.join(m['content'] for m in current_group)
                    cleaned_content = self.clean_text(combined_content)
                    if cleaned_content.strip():  # 정제된 내용이 있는 경우만 추가
                        grouped.append({
                            'username': current_user,
                            'time': current_group[0]['time'],
                            'content': cleaned_content
                        })
                current_group = [msg]
                current_user = msg['username']
            else:
                current_group.append(msg)

        if current_group:
            combined_content = ' '.join(m['content'] for m in current_group)
            cleaned_content = self.clean_text(combined_content)
            if cleaned_content.strip():
                grouped.append({
                    'username': current_user,
                    'time': current_group[0]['time'],
                    'content': cleaned_content
                })

        return grouped

    def extract_key_points(self, messages: List[Dict]) -> List[str]:
        """대화 내용에서 주요 포인트 추출"""
        key_points = []
        current_topic = []
        
        for msg in messages:
            content = self.clean_text(msg['content'])
            if not content:  # 정제 후 내용이 없으면 건너뛰기
                continue
                
            current_topic.append(content)
            
            # 일정 길이 이상 모였거나 마지막 메시지인 경우
            if len(current_topic) >= 3 or msg == messages[-1]:
                # 주요 내용 추출
                if current_topic:
                    combined = ' '.join(current_topic)
                    # 중복 내용 제거
                    combined = ' '.join(dict.fromkeys(combined.split()))
                    # 긴 내용은 요약
                    if len(combined) > 50:
                        combined = combined[:47] + '...'
                    if combined.strip():  # 의미 있는 내용만 추가
                        key_points.append(combined)
                current_topic = []
        
        return key_points

    def identify_themes(self, messages: List[Dict], num_themes: int = 5) -> List[Dict]:
        """대화 테마 식별"""
        # 메시지 그룹화
        grouped_messages = self.group_consecutive_messages(messages)
        
        # 전처리된 텍스트 준비
        preprocessed_texts = [self.preprocess_text(msg['content']) 
                            for msg in grouped_messages]
        
        # 빈 문자열 필터링
        valid_indices = [i for i, text in enumerate(preprocessed_texts) if text.strip()]
        preprocessed_texts = [preprocessed_texts[i] for i in valid_indices]
        valid_messages = [grouped_messages[i] for i in valid_indices]
        
        if not preprocessed_texts:
            return []

        # TF-IDF 벡터화
        vectorizer = TfidfVectorizer(max_features=1000)
        try:
            tfidf_matrix = vectorizer.fit_transform(preprocessed_texts)
        except ValueError:
            return []

        # 실제 테마 수 조정
        num_themes = min(num_themes, len(preprocessed_texts))
        if num_themes < 2:
            num_themes = 2

        # K-means 클러스터링
        kmeans = KMeans(n_clusters=num_themes, random_state=42)
        clusters = kmeans.fit_predict(tfidf_matrix)

        # 테마별 메시지 그룹화
        themes = defaultdict(list)
        for idx, cluster in enumerate(clusters):
            themes[cluster].append(valid_messages[idx])

        return [{'theme_id': i, 'messages': msgs} 
                for i, msgs in themes.items()]

    def summarize_themes(self, chat_text: str) -> str:
        """채팅 내용을 테마별로 요약"""
        # 채팅 파싱
        messages = self.parse_chat(chat_text)
        if not messages:
            return "분석할 채팅 내용이 없습니다."

        # 테마 식별
        themes = self.identify_themes(messages)
        if not themes:
            return "테마를 식별할 수 없습니다."

        # 결과 포맷팅
        summary = "[테마별 대화 요약]\n\n"
        
        for theme in themes:
            summary += f"{theme['theme_id'] + 1}.\n"
            
            # 주요 포인트 추출 및 정제
            key_points = self.extract_key_points(theme['messages'])
            
            # 중복 제거 및 정렬
            unique_points = []
            for point in key_points:
                cleaned_point = point.strip()
                if cleaned_point and cleaned_point not in unique_points:
                    unique_points.append(cleaned_point)
            
            # 개조식으로 정리
            for point in unique_points:
                summary += f"- {point}\n"
            
            summary += "\n"
        
        return summary

class AnalyzerThread(QThread):
    """분석 작업을 위한 스레드"""
    finished = pyqtSignal(str)
    progress = pyqtSignal(int)

    def __init__(self, analyzer, chat_text):
        super().__init__()
        self.analyzer = analyzer
        self.chat_text = chat_text

    def run(self):
        try:
            # 진행률 업데이트
            self.progress.emit(30)
            result = self.analyzer.summarize_themes(self.chat_text)
            self.progress.emit(100)
            self.finished.emit(result)
        except Exception as e:
            self.finished.emit(f"분석 중 오류 발생: {str(e)}")

class ChatAnalyzerGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.analyzer = ChatThemeAnalyzer()
        self.init_ui()

    def init_ui(self):
        """UI 초기화"""
        self.setWindowTitle('채팅 테마 분석기')
        self.setGeometry(100, 100, 1200, 800)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f0f0;
            }
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:pressed {
                background-color: #0D47A1;
            }
            QTextEdit {
                background-color: white;
                border: 1px solid #ddd;
                border-radius: 4px;
                padding: 8px;
                color: black;  /* 텍스트 색상을 검정색으로 변경 */
            }
            QLabel {
                color: black;  /* 라벨 텍스트 색상을 검정색으로 변경 */
                font-weight: bold;
            }
            QProgressBar {
                border: 2px solid #2196F3;
                border-radius: 5px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #2196F3;
            }
        """)

        # 메인 위젯 및 레이아웃
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout()
        main_widget.setLayout(layout)

        # 좌측 패널 (입력)
        left_panel = QWidget()
        left_layout = QVBoxLayout()
        left_panel.setLayout(left_layout)

        # 우측 패널 (출력)
        right_panel = QWidget()
        right_layout = QVBoxLayout()
        right_panel.setLayout(right_layout)

        # 입력 영역
        input_label = QLabel('채팅 내용 입력:')
        self.input_text = QTextEdit()
        self.input_text.setPlaceholderText("채팅 로그를 여기에 붙여넣으세요...")

        # 버튼 영역
        button_layout = QHBoxLayout()
        self.load_button = QPushButton('파일 불러오기')
        self.analyze_button = QPushButton('분석 시작')
        self.save_button = QPushButton('결과 저장')
        self.clear_button = QPushButton('초기화')
        
        button_layout.addWidget(self.load_button)
        button_layout.addWidget(self.analyze_button)
        button_layout.addWidget(self.save_button)
        button_layout.addWidget(self.clear_button)

        # 진행바
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximum(100)
        self.progress_bar.setTextVisible(True)

        # 출력 영역
        output_label = QLabel('분석 결과:')
        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)

        # 좌측 패널에 위젯 추가
        left_layout.addWidget(input_label)
        left_layout.addWidget(self.input_text)
        left_layout.addLayout(button_layout)
        left_layout.addWidget(self.progress_bar)

        # 우측 패널에 위젯 추가
        right_layout.addWidget(output_label)
        right_layout.addWidget(self.output_text)

        # 패널을 메인 레이아웃에 추가
        layout.addWidget(left_panel)
        layout.addWidget(right_panel)

        # 이벤트 연결
        self.load_button.clicked.connect(self.load_file)
        self.analyze_button.clicked.connect(self.analyze_chat)
        self.save_button.clicked.connect(self.save_result)
        self.clear_button.clicked.connect(self.clear_all)

        # 분석 스레드 초기화
        self.analyzer_thread = None

    def load_file(self):
        """파일 불러오기"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, '채팅 로그 파일 선택', '', 
            'Text Files (*.txt);;All Files (*.*)'
        )
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    self.input_text.setText(file.read())
                self.show_message('성공', '파일을 성공적으로 불러왔습니다.')
            except Exception as e:
                self.show_message('오류', f'파일 불러오기 실패: {str(e)}')

    def analyze_chat(self):
        """채팅 분석 시작"""
        chat_text = self.input_text.toPlainText()
        if not chat_text.strip():
            self.show_message('경고', '분석할 채팅 내용을 입력해주세요.')
            return

        # 버튼 비활성화
        self.analyze_button.setEnabled(False)
        self.progress_bar.setValue(0)

        # 분석 스레드 시작
        self.analyzer_thread = AnalyzerThread(self.analyzer, chat_text)
        self.analyzer_thread.finished.connect(self.analysis_finished)
        self.analyzer_thread.progress.connect(self.progress_bar.setValue)
        self.analyzer_thread.start()

    def analysis_finished(self, result):
        """분석 완료 후 처리"""
        self.output_text.setText(result)
        self.analyze_button.setEnabled(True)
        self.progress_bar.setValue(100)
        self.show_message('완료', '분석이 완료되었습니다.')

    def save_result(self):
        """결과 저장"""
        result = self.output_text.toPlainText()
        if not result:
            self.show_message('경고', '저장할 분석 결과가 없습니다.')
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, '분석 결과 저장', 
            f'chat_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt',
            'Text Files (*.txt);;All Files (*.*)'
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as file:
                    file.write(result)
                self.show_message('성공', '결과가 성공적으로 저장되었습니다.')
            except Exception as e:
                self.show_message('오류', f'저장 실패: {str(e)}')

    def clear_all(self):
        """모든 입력과 출력 초기화"""
        self.input_text.clear()
        self.output_text.clear()
        self.progress_bar.setValue(0)

    def show_message(self, title, message):
        """메시지 박스 표시"""
        QMessageBox.information(self, title, message)

def main():
    app = QApplication(sys.argv)
    
    # 스타일 설정
    app.setStyle('Fusion')
    
    # 라이트 모드 팔레트 설정
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(240, 240, 240))
    palette.setColor(QPalette.WindowText, Qt.black)
    palette.setColor(QPalette.Base, Qt.white)
    palette.setColor(QPalette.AlternateBase, QColor(245, 245, 245))
    palette.setColor(QPalette.ToolTipBase, Qt.white)
    palette.setColor(QPalette.ToolTipText, Qt.black)
    palette.setColor(QPalette.Text, Qt.black)
    palette.setColor(QPalette.Button, QColor(240, 240, 240))
    palette.setColor(QPalette.ButtonText, Qt.black)
    palette.setColor(QPalette.BrightText, Qt.red)
    palette.setColor(QPalette.Link, QColor(0, 0, 255))
    palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.HighlightedText, Qt.black)
    
    app.setPalette(palette)
    
    ex = ChatAnalyzerGUI()
    ex.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()