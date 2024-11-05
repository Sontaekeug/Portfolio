import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, Trainer, TrainingArguments
import tkinter as tk
from tkinter import scrolledtext, messagebox, ttk
from datasets import Dataset
import threading
import queue
import logging
import os

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='chatbot.log'
)

class QAChatbot:
    def __init__(self):
        self.model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"
        self.train_data = []
        self.setup_model()
        self.setup_gui()
        self.task_queue = queue.Queue()

    def setup_model(self):
        """모델과 토크나이저 초기화"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForQuestionAnswering.from_pretrained(self.model_name)
            logging.info("모델 로딩 완료")
        except Exception as e:
            logging.error(f"모델 로딩 실패: {str(e)}")
            raise

    def setup_gui(self):
        """GUI 초기화 및 구성"""
        self.window = tk.Tk()
        self.window.title("Q&A Chatbot")
        self.window.geometry("600x600")
        
        # 스타일 설정
        style = ttk.Style()
        style.configure("TButton", padding=5)
        style.configure("TLabel", padding=5)
        
        # 프레임 생성
        input_frame = ttk.Frame(self.window, padding="10")
        input_frame.pack(fill=tk.BOTH, expand=True)
        
        # GUI 요소 생성
        self.create_gui_elements(input_frame)
        
        # 상태바 추가
        self.status_var = tk.StringVar()
        self.status_var.set("준비됨")
        self.status_bar = ttk.Label(self.window, textvariable=self.status_var, relief=tk.SUNKEN)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def create_gui_elements(self, parent):
        """GUI 요소 생성 및 배치"""
        # 질문 입력
        ttk.Label(parent, text="질문:").pack(anchor='w')
        self.entry_question = scrolledtext.ScrolledText(parent, height=3, width=60)
        self.entry_question.pack(fill=tk.X, pady=5)

        # 컨텍스트 입력
        ttk.Label(parent, text="배경지식:").pack(anchor='w')
        self.entry_context = scrolledtext.ScrolledText(parent, height=6, width=60)
        self.entry_context.pack(fill=tk.X, pady=5)

        # 답변 출력
        ttk.Label(parent, text="답변:").pack(anchor='w')
        self.entry_answer = scrolledtext.ScrolledText(parent, height=3, width=60)
        self.entry_answer.pack(fill=tk.X, pady=5)

        # 버튼 프레임
        button_frame = ttk.Frame(parent)
        button_frame.pack(fill=tk.X, pady=10)

        # 버튼 생성
        ttk.Button(button_frame, text="답변 생성", command=self.generate_answer_thread).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="학습 예제 추가", command=self.add_training_example).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="모델 학습", command=self.train_model_thread).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="초기화", command=self.clear_fields).pack(side=tk.LEFT, padx=5)

    def get_answer(self, question, context):
        """답변 생성"""
        try:
            inputs = self.tokenizer(question, context, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # 답변 추출 및 후처리
            answer_start = torch.argmax(outputs.start_logits)
            answer_end = torch.argmax(outputs.end_logits) + 1
            
            if answer_end <= answer_start:
                return "유효한 답변을 찾을 수 없습니다."
                
            answer = self.tokenizer.convert_tokens_to_string(
                self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end])
            )
            
            return answer.strip() if answer.strip() else "답변을 찾을 수 없습니다."
            
        except Exception as e:
            logging.error(f"답변 생성 중 오류 발생: {str(e)}")
            return f"오류가 발생했습니다: {str(e)}"

    def generate_answer_thread(self):
        """답변 생성을 위한 스레드 실행"""
        question = self.entry_question.get("1.0", "end-1c").strip()
        context = self.entry_context.get("1.0", "end-1c").strip()
        
        if not question or not context:
            messagebox.showerror("오류", "질문과 컨텍스트를 모두 입력해주세요.")
            return
            
        self.status_var.set("답변 생성 중...")
        thread = threading.Thread(target=self._generate_answer_task, args=(question, context))
        thread.daemon = True
        thread.start()

    def _generate_answer_task(self, question, context):
        """실제 답변 생성 작업"""
        try:
            answer = self.get_answer(question, context)
            self.window.after(0, self._update_answer, answer)
        except Exception as e:
            self.window.after(0, self._show_error, str(e))
        finally:
            self.window.after(0, lambda: self.status_var.set("준비됨"))

    def _update_answer(self, answer):
        """UI에 답변 업데이트"""
        self.entry_answer.delete("1.0", tk.END)
        self.entry_answer.insert(tk.END, answer)

    def _show_error(self, error_message):
        """에러 메시지 표시"""
        messagebox.showerror("오류", f"답변 생성 중 오류가 발생했습니다: {error_message}")

    def add_training_example(self):
        """학습 예제 추가"""
        question = self.entry_question.get("1.0", "end-1c").strip()
        context = self.entry_context.get("1.0", "end-1c").strip()
        answer = self.entry_answer.get("1.0", "end-1c").strip()
        
        if not all([question, context, answer]):
            messagebox.showerror("오류", "질문, 컨텍스트, 답변을 모두 입력해주세요.")
            return
            
        try:
            answer_start = context.find(answer)
            if answer_start == -1:
                messagebox.showerror("오류", "답변이 컨텍스트 내에 존재하지 않습니다.")
                return
                
            self.train_data.append({
                "question": question,
                "context": context,
                "answers": {"text": [answer], "answer_start": [answer_start]}
            })
            messagebox.showinfo("성공", "학습 예제가 추가되었습니다.")
            logging.info(f"학습 예제 추가됨: {question}")
            
        except Exception as e:
            logging.error(f"학습 예제 추가 중 오류: {str(e)}")
            messagebox.showerror("오류", f"학습 예제 추가 중 오류가 발생했습니다: {str(e)}")

    def train_model_thread(self):
        """모델 학습을 위한 스레드 실행"""
        if not self.train_data:
            messagebox.showerror("오류", "학습할 데이터가 없습니다.")
            return
            
        self.status_var.set("모델 학습 중...")
        thread = threading.Thread(target=self._train_model_task)
        thread.daemon = True
        thread.start()

    def _train_model_task(self):
        """실제 모델 학습 작업"""
        try:
            dataset = Dataset.from_dict({
                "question": [d["question"] for d in self.train_data],
                "context": [d["context"] for d in self.train_data],
                "answers": [d["answers"] for d in self.train_data]
            })
            
            training_args = TrainingArguments(
                output_dir="./results",
                num_train_epochs=3,
                per_device_train_batch_size=4,
                logging_dir="./logs",
                logging_steps=10,
                save_strategy="epoch"
            )
            
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=dataset
            )
            
            trainer.train()
            self.window.after(0, lambda: messagebox.showinfo("성공", "모델 학습이 완료되었습니다."))
            logging.info("모델 학습 완료")
            
        except Exception as e:
            logging.error(f"모델 학습 중 오류: {str(e)}")
            self.window.after(0, lambda: messagebox.showerror("오류", f"모델 학습 중 오류가 발생했습니다: {str(e)}"))
        finally:
            self.window.after(0, lambda: self.status_var.set("준비됨"))

    def clear_fields(self):
        """입력 필드 초기화"""
        self.entry_question.delete("1.0", tk.END)
        self.entry_context.delete("1.0", tk.END)
        self.entry_answer.delete("1.0", tk.END)

    def run(self):
        """애플리케이션 실행"""
        try:
            self.window.mainloop()
        except Exception as e:
            logging.error(f"애플리케이션 실행 중 오류: {str(e)}")
            raise

if __name__ == "__main__":
    try:
        app = QAChatbot()
        app.run()
    except Exception as e:
        logging.error(f"프로그램 초기화 중 오류: {str(e)}")
        messagebox.showerror("치명적 오류", f"프로그램을 시작할 수 없습니다: {str(e)}")