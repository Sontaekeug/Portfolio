import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QTextEdit, QPushButton, QLabel
from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration
import torch

class TextSummarizerGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('텍스트 요약 프로그램')
        
        # KoBART 모델과 토크나이저 로드
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained('gogamza/kobart-base-v2')
        self.model = BartForConditionalGeneration.from_pretrained('gogamza/kobart-base-v2')
        
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
        
    def summarize_text(self):
        input_text = self.input_text.toPlainText()
        if not input_text:
            self.output_text.setText("텍스트를 입력해주세요.")
            return
        
        try:
            # 텍스트 토큰화 및 요약 생성 부분은 동일
            inputs = self.tokenizer(input_text, 
                                  max_length=1024, 
                                  truncation=True, 
                                  return_tensors='pt')
            
            with torch.no_grad():
                summary_ids = self.model.generate(
                    inputs['input_ids'],
                    max_length=150,
                    min_length=50,
                    length_penalty=2.0,
                    num_beams=4,
                    repetition_penalty=2.0,
                    early_stopping=True
                )
            
            summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            
            # 요약문을 문장 단위로 분리하고 주제어 추출하여 포맷팅
            sentences = summary.split('. ')
            formatted_summary = []
            
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                
                # 주제어 추출 로직
                words = sentence.split()
                if len(words) < 2:
                    continue
                    
                # 첫 2-3개 단어를 기반으로 주제어 추출
                topic_words = words[:min(3, len(words))]
                topic = ' '.join(topic_words)
                
                # 주제어에서 조사 제거
                topic = topic.rstrip('이가은는을를의에')
                
                # 문장에서 주제어 부분 제거하여 중복 방지
                content = sentence
                if content.startswith(topic):
                    content = content[len(topic):].lstrip('은는이가을를에서')
                
                # 최종 포맷팅
                formatted_line = f"• {topic}: {content.strip()}"
                formatted_summary.append(formatted_line)
            
            # 최종 요약문 생성
            final_summary = "\n\n".join(formatted_summary)
            
            if not final_summary.strip():
                self.output_text.setText("요약을 생성할 수 없습니다.")
                return
                
            self.output_text.setText(final_summary)
            
        except Exception as e:
            self.output_text.setText(f"요약 중 오류가 발생했습니다: {str(e)}")

def main():
    app = QApplication(sys.argv)
    window = TextSummarizerGUI()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()