import sys
import os
import torch  # PyTorch 라이브러리: 딥러닝 모델 구현 및 실행
import cv2  # OpenCV: 이미지 처리 및 컴퓨터 비전
import numpy as np  # NumPy: 배열 및 수학적 연산
from PIL import Image  # PIL: 이미지 파일 처리
from torchvision import transforms  # PyTorch 도구: 이미지 전처리 및 변환
from torch import nn  # PyTorch 신경망 모듈
import torch.nn.functional as F  # PyTorch 활성화 함수 및 기타 유틸리티
from pathlib import Path  # 파일 및 디렉터리 작업
import tkinter as tk  # Tkinter: GUI 생성
from tkinter import ttk, filedialog, messagebox  # Tkinter 위젯 및 파일 대화 상자
from PIL import Image, ImageTk  # Tkinter용 이미지 처리

# 딥러닝 모델 정의: CNN(합성곱 신경망) 기반
class PetCNN(nn.Module):
    def __init__(self):
        super(PetCNN, self).__init__()
        # 합성곱 계층 및 풀링 계층 정의
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)  # 3채널 입력, 16채널 출력, 필터 크기 3x3
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)  # 최대 풀링 계층: 2x2 윈도우
        # 완전 연결 계층 정의
        self.fc1 = nn.Linear(64 * 28 * 28, 512)  # 첫 번째 완전 연결 계층
        self.fc2 = nn.Linear(512, 2)  # 최종 출력 (2개 클래스: 고양이, 강아지)
        self.dropout = nn.Dropout(0.25)  # 드롭아웃: 과적합 방지

    def forward(self, x):
        # 순전파 구현: 계층을 통과하며 데이터 처리
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 28 * 28)  # 텐서 차원 변환 (배치 크기 유지)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# 반려동물 탐지 클래스 정의
class PetDetector:
    def __init__(self):
        # YOLOv5 모델 불러오기 (객체 탐지)
        self.yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5m')  
        self.yolo_model.classes = [15, 16]  # 고양이(15)와 강아지(16)에 초점
        self.yolo_model.conf = 0.4  # 신뢰도 임계값 설정
        
        # CNN 모델 초기화 및 가중치 불러오기
        self.cnn_model = PetCNN()
        if Path('pet_classifier.pth').exists():
            self.cnn_model.load_state_dict(torch.load('pet_classifier.pth'))
        self.cnn_model.eval()  # 모델을 평가 모드로 전환
        
        # 이미지 전처리를 위한 변환 정의
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # 이미지 크기 조정
            transforms.ToTensor(),  # 텐서로 변환
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 정규화
        ])
        self.classes = ['cat', 'dog']  # 클래스 레이블

    def process_image(self, image_path):
        # 이미지 처리 및 탐지 함수
        try:
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError("이미지를 읽을 수 없습니다")

            results = self.yolo_model(img)  # YOLO 모델로 탐지 수행
            
            detections = []  # 탐지된 결과 저장
            for *xyxy, conf, cls in results.xyxy[0]:  # 탐지된 객체 루프
                if conf > 0.4:  # 신뢰도 기준 통과
                    x1, y1, x2, y2 = map(int, xyxy)  # 경계 상자 좌표
                    original_class = 'cat' if int(cls) == 15 else 'dog'
                    
                    # 이미지 경계 확인 및 잘라내기
                    height, width = img.shape[:2]
                    x1, x2 = max(0, x1), min(width, x2)
                    y1, y2 = max(0, y1), min(height, y2)
                    
                    if x2 > x1 and y2 > y1:  # 유효 영역 확인
                        cropped_img = img[y1:y2, x1:x2].copy()
                        if cropped_img.size > 0:
                            try:
                                classification = self.classify_image(cropped_img)
                                final_class = self.combine_predictions(
                                    original_class, 
                                    classification['class'], 
                                    float(conf), 
                                    classification['confidence']
                                )
                                
                                detections.append({
                                    'class': final_class['class'],
                                    'confidence': final_class['confidence'],
                                    'bbox': (x1, y1, x2, y2)
                                })
                                
                                # 경계 상자 및 레이블 그리기
                                color = (0, 255, 0) if final_class['class'] == 'dog' else (0, 0, 255)
                                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                                label = f"{final_class['class']} {final_class['confidence']:.2f}"
                                cv2.putText(img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                            except Exception as e:
                                print(f"개별 객체 처리 중 오류: {str(e)}")
                                continue
            
            return detections, img
        except Exception as e:
            print(f"이미지 처리 중 오류: {str(e)}")
            raise

    def combine_predictions(self, yolo_class, cnn_class, yolo_conf, cnn_conf):
        # YOLO와 CNN의 예측 결과를 결합하여 최종 클래스 및 신뢰도를 결정
        if yolo_class == cnn_class:
            # 두 모델의 결과가 일치하는 경우 평균 신뢰도를 반환
            return {
                'class': yolo_class,
                'confidence': (yolo_conf + cnn_conf) / 2
            }
        
        # 결과가 불일치하는 경우, 더 높은 신뢰도를 가진 모델의 결과를 선택
        if yolo_conf > cnn_conf:
            return {
                'class': yolo_class,
                'confidence': yolo_conf
            }
        else:
            return {
                'class': cnn_class,
                'confidence': cnn_conf
            }

    def classify_image(self, image):
        # 이미지를 분류하기 위해 CNN 모델에 전달
        if isinstance(image, np.ndarray):
            # OpenCV 이미지를 PIL 이미지로 변환
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
        # 이미지 전처리
        img_tensor = self.transform(image).unsqueeze(0)  # 배치 차원을 추가
        
        with torch.no_grad():  # 그래디언트 비활성화 (추론 단계에서 필요 없음)
            outputs = self.cnn_model(img_tensor)  # CNN 모델에 이미지 전달
            probabilities = F.softmax(outputs, dim=1)[0]  # 확률 값 계산
            
            # 고양이와 강아지의 확률 추출
            cat_prob = probabilities[0].item()
            dog_prob = probabilities[1].item()
            
            if abs(cat_prob - dog_prob) < 0.2:  # 신뢰도가 낮은 경우
                return {
                    'class': 'unknown',
                    'confidence': max(cat_prob, dog_prob)
                }
            
            # 더 높은 확률을 가진 클래스를 선택
            if cat_prob > dog_prob:
                return {
                    'class': 'cat',
                    'confidence': cat_prob
                }
            else:
                return {
                    'class': 'dog',
                    'confidence': dog_prob
                }

# GUI 클래스 정의: Tkinter를 사용하여 사용자 인터페이스 구현
class PetDetectorGUI:
    def __init__(self, root):
        # Tkinter 윈도우 설정
        self.root = root
        self.root.title("반려동물 탐지 프로그램")
        self.root.geometry("1200x800")
        
        # 반려동물 탐지기 초기화
        self.detector = PetDetector()
        
        self.setup_gui()  # GUI 초기화 메서드 호출
        
    def setup_gui(self):
        # 메인 프레임 생성
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 이미지 표시 영역 생성
        self.canvas = tk.Canvas(main_frame, width=800, height=600)
        self.canvas.grid(row=0, column=0, rowspan=6, padx=5, pady=5)
        
        # 컨트롤 프레임 생성
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 버튼 추가
        ttk.Button(control_frame, text="이미지 선택", command=self.load_image).grid(row=0, column=0, pady=5)
        ttk.Button(control_frame, text="탐지 시작", command=self.detect_pets).grid(row=1, column=0, pady=5)
        ttk.Button(control_frame, text="저장", command=self.save_image).grid(row=2, column=0, pady=5)
        
        # 결과 표시 영역 생성
        self.result_text = tk.Text(control_frame, width=40, height=20)
        self.result_text.grid(row=3, column=0, pady=5)
        
        # 현재 및 처리된 이미지 초기화
        self.current_image = None
        self.processed_image = None
        
    def load_image(self):
        # 파일 대화상자를 사용해 이미지 파일 선택
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff")]
        )
        if file_path:
            self.current_image_path = file_path
            image = Image.open(file_path)  # 이미지 열기
            image.thumbnail((800, 600))  # 썸네일 크기로 조정
            photo = ImageTk.PhotoImage(image)
            
            # 캔버스에 이미지 표시
            self.canvas.config(width=photo.width(), height=photo.height())
            self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
            self.canvas.image = photo
            
            # 결과 텍스트 초기화
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, "이미지가 로드되었습니다.\n")
    
    def detect_pets(self):
        # 탐지를 시작하는 메서드
        if not hasattr(self, 'current_image_path'):
            messagebox.showerror("오류", "먼저 이미지를 선택해주세요!")
            return
            
        try:
            detections, processed_img = self.detector.process_image(self.current_image_path)
            
            # 탐지 결과 표시
            self.result_text.delete(1.0, tk.END)
            if not detections:
                self.result_text.insert(tk.END, "탐지된 반려동물이 없습니다.\n")
            else:
                for i, det in enumerate(detections, 1):
                    self.result_text.insert(tk.END, f"탐지 {i}:\n")
                    self.result_text.insert(tk.END, f"종류: {det['class']}\n")
                    self.result_text.insert(tk.END, f"신뢰도: {det['confidence']:.2f}\n")
                    self.result_text.insert(tk.END, "---\n")
            
            # 처리된 이미지 표시
            if processed_img is not None:
                processed_img_rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(processed_img_rgb)
                image.thumbnail((800, 600))
                photo = ImageTk.PhotoImage(image)
                
                self.canvas.config(width=photo.width(), height=photo.height())
                self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
                self.canvas.image = photo
                self.processed_image = processed_img
            
        except Exception as e:
            print(f"에러 상세 정보: {str(e)}")
            messagebox.showerror("오류", f"이미지 처리 중 오류가 발생했습니다:\n{str(e)}")
    
    def save_image(self):
        # 처리된 이미지를 저장하는 메서드
        if self.processed_image is None:
            messagebox.showerror("오류", "저장할 이미지가 없습니다!")
            return
            
        file_path = filedialog.asksaveasfilename(
            defaultextension=".jpg",
            filetypes=[("JPEG files", "*.jpg"), ("PNG files", "*.png")]
        )
        if file_path:
            cv2.imwrite(file_path, self.processed_image)  # 이미지 파일 저장
            messagebox.showinfo("성공", "이미지가 저장되었습니다!")

# 프로그램 진입점
def main():
    root = tk.Tk()  # Tkinter 윈도우 생성
    app = PetDetectorGUI(root)  # GUI 애플리케이션 생성
    root.mainloop()  # 이벤트 루프 시작

# 메인 프로그램 실행
if __name__ == "__main__":
    main()
