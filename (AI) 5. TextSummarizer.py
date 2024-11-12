import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QTextEdit, QPushButton, QLabel
from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration
import torch

class TextSummarizer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('텍스트 요약 프로그램')
        
        # KoBART 모델과 토크나이저 로드 - 요약 전용 모델로 변경
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained('gogamza/kobart-summarization')
        self.model = BartForConditionalGeneration.from_pretrained('gogamza/kobart-summarization')
        
        # 중앙 위젯 생성
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # 입력 텍스트 영역
        input_label = QLabel('원본 텍스트:')
        layout.addWidget(input_label)
        self.input_text = QTextEdit()
        layout.addWidget(self.input_text)
        
        # 요약 버튼
        self.summarize_button = QPushButton('텍스트 요약하기')
        self.summarize_button.clicked.connect(self.summarize_text)
        layout.addWidget(self.summarize_button)
        
        # 출력 텍스트 영역
        output_label = QLabel('요약 결과:')
        layout.addWidget(output_label)
        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        layout.addWidget(self.output_text)
        
    def preprocess_text(self, text):
        # 개선된 텍스트 전처리
        text = ' '.join(text.split())
        # 연속된 마침표 정리
        text = text.replace('...', '.')
        text = text.replace('..', '.')
        # 불필요한 공백 제거
        text = text.replace(' .', '.')
        text = text.replace('\n', ' ')
        return text
        
    def summarize_text(self):
        input_text = self.input_text.toPlainText()
        if not input_text:
            self.output_text.setText("텍스트를 입력해주세요.")
            return
        
        try:
            processed_text = self.preprocess_text(input_text)
            
            inputs = self.tokenizer(processed_text,
                                  max_length=2048,  # 입력 길이 증가
                                  truncation=True,
                                  padding=True,
                                  return_tensors='pt')
            
            summary_ids = self.model.generate(
                inputs['input_ids'],
                max_length=512,             # 요약 최대 길이 대폭 증가
                min_length=150,             # 최소 길이 증가
                length_penalty=1.2,         # 길이 페널티 감소로 더 긴 요약 유도
                num_beams=8,                # 빔 서치 개수 증가
                early_stopping=False,       # 조기 종료 비활성화
                top_p=0.95,                 # 다양성 증가
                top_k=120,                  # 선택 토큰 수 증가
                repetition_penalty=1.8,     # 반복 페널티 완화
                no_repeat_ngram_size=2,     # n-gram 반복 제한 완화
                temperature=0.8             # 다양성 약간 증가
            )
            
            summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            
            # 요약문 후처리 및 포맷팅
            summary_sentences = [s.strip() for s in summary.split('.') if len(s.strip()) > 8]
            formatted_summary = '\n'.join(f'- {sentence}.' for sentence in summary_sentences)
            
            self.output_text.setText(formatted_summary)
            
        except Exception as e:
            self.output_text.setText(f"요약 중 오류가 발생했습니다: {str(e)}")

def main():
    app = QApplication(sys.argv)
    window = TextSummarizer()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()