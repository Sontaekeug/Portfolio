import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QTextEdit, QPushButton, QLabel
# PyQt5: GUI 애플리케이션 구성 요소 (창, 버튼, 텍스트 박스 등)
from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration
# Transformers: KoBART 모델과 토크나이저를 사용하기 위한 라이브러리
import torch
# PyTorch: KoBART 모델 실행 및 텐서 연산

class TextSummarizer(QMainWindow):
    """
    텍스트 요약 GUI 프로그램 클래스
    - PyQt5를 사용하여 GUI를 생성
    - KoBART 모델을 활용하여 입력된 텍스트를 요약
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle('텍스트 요약 프로그램')  # 창 제목 설정
        
        # KoBART 모델과 토크나이저 초기화
        # 사전 학습된 요약 모델과 토크나이저를 로드
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained('gogamza/kobart-summarization')
        self.model = BartForConditionalGeneration.from_pretrained('gogamza/kobart-summarization')
        
        # 중앙 위젯과 레이아웃 설정
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # 입력 텍스트 영역
        input_label = QLabel('원본 텍스트:')
        layout.addWidget(input_label)
        self.input_text = QTextEdit()  # 입력 텍스트를 위한 QTextEdit
        layout.addWidget(self.input_text)
        
        # 요약 버튼
        self.summarize_button = QPushButton('텍스트 요약하기')
        self.summarize_button.clicked.connect(self.summarize_text)  # 버튼 클릭 시 요약 실행
        layout.addWidget(self.summarize_button)
        
        # 출력 텍스트 영역
        output_label = QLabel('요약 결과:')
        layout.addWidget(output_label)
        self.output_text = QTextEdit()  # 요약 결과를 보여줄 QTextEdit
        self.output_text.setReadOnly(True)  # 결과 텍스트는 읽기 전용으로 설정
        layout.addWidget(self.output_text)
        
    def preprocess_text(self, text):
        """
        입력 텍스트를 요약에 적합한 형태로 전처리
        - 공백, 불필요한 마침표 등을 정리
        """
        text = ' '.join(text.split())  # 여러 줄의 공백을 제거
        text = text.replace('...', '.').replace('..', '.')  # 연속된 마침표 처리
        text = text.replace(' .', '.').replace('\n', ' ')  # 불필요한 공백과 줄바꿈 제거
        return text
        
    def summarize_text(self):
        """
        텍스트 요약 실행
        - 입력 텍스트를 전처리
        - KoBART 모델을 사용하여 요약 생성
        - 결과를 GUI에 출력
        """
        input_text = self.input_text.toPlainText()  # 입력된 텍스트 가져오기
        if not input_text:  # 입력 텍스트가 없는 경우
            self.output_text.setText("텍스트를 입력해주세요.")
            return
        
        try:
            # 입력 텍스트 전처리
            processed_text = self.preprocess_text(input_text)
            
            # 텍스트를 토큰화하여 모델 입력으로 변환
            inputs = self.tokenizer(processed_text,
                                    max_length=2048,  # 최대 입력 길이
                                    truncation=True,  # 초과 부분 제거
                                    padding=True,  # 패딩 추가
                                    return_tensors='pt')  # PyTorch 텐서로 반환
            
            # KoBART 모델을 사용하여 요약 생성
            summary_ids = self.model.generate(
                inputs['input_ids'],
                max_length=512,             # 요약문 최대 길이
                min_length=150,             # 요약문 최소 길이
                length_penalty=1.2,         # 길이에 대한 페널티 조정
                num_beams=8,                # 빔 서치 크기
                early_stopping=False,       # 조기 종료 비활성화
                top_p=0.95,                 # 확률 기반 샘플링
                top_k=120,                  # 상위 K개 토큰 선택
                repetition_penalty=1.8,     # 반복 페널티
                no_repeat_ngram_size=2,     # 반복 방지 n-gram 크기
                temperature=0.8             # 출력 다양성 조정
            )
            
            # 모델 출력 디코딩 및 후처리
            summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            summary_sentences = [s.strip() for s in summary.split('.') if len(s.strip()) > 8]  # 간결한 문장 필터링
            formatted_summary = '\n'.join(f'- {sentence}.' for sentence in summary_sentences)  # 포맷팅
            
            # 결과를 출력 텍스트 영역에 표시
            self.output_text.setText(formatted_summary)
            
        except Exception as e:
            # 요약 중 오류가 발생하면 메시지 출력
            self.output_text.setText(f"요약 중 오류가 발생했습니다: {str(e)}")

def main():
    """
    프로그램의 진입점
    - PyQt5 애플리케이션 초기화
    - TextSummarizer 창을 생성하고 실행
    """
    app = QApplication(sys.argv)  # QApplication 생성
    window = TextSummarizer()  # TextSummarizer GUI 창 생성
    window.show()  # 창 표시
    sys.exit(app.exec_())  # 애플리케이션 실행

if __name__ == '__main__':
    main()  # 프로그램 실행
