import sys
import os
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from torch import nn
import torch.nn.functional as F
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk

class PetCNN(nn.Module):
    def __init__(self):
        super(PetCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, 2)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 28 * 28)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

class PetDetector:
    def __init__(self):
        self.yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5m')  # 더 큰 모델 사용
        self.yolo_model.classes = [15, 16]  # COCO dataset: cat(15), dog(16)
        self.yolo_model.conf = 0.4  # 적절한 신뢰도 임계값
        
        self.cnn_model = PetCNN()
        if Path('pet_classifier.pth').exists():
            self.cnn_model.load_state_dict(torch.load('pet_classifier.pth'))
        self.cnn_model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
        ])
        
        self.classes = ['cat', 'dog']

    def process_image(self, image_path):
        try:
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError("이미지를 읽을 수 없습니다")

            results = self.yolo_model(img)
            
            detections = []
            for *xyxy, conf, cls in results.xyxy[0]:
                if conf > 0.4:
                    x1, y1, x2, y2 = map(int, xyxy)
                    original_class = 'cat' if int(cls) == 15 else 'dog'
                    
                    # 이미지 경계 확인
                    height, width = img.shape[:2]
                    x1, x2 = max(0, x1), min(width, x2)
                    y1, y2 = max(0, y1), min(height, y2)
                    
                    if x2 > x1 and y2 > y1:  # 유효한 영역인지 확인
                        cropped_img = img[y1:y2, x1:x2].copy()  # 이미지 복사본 사용
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
                                
                                color = (0, 255, 0) if final_class['class'] == 'dog' else (0, 0, 255)
                                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                                label = f"{final_class['class']} {final_class['confidence']:.2f}"
                                cv2.putText(img, label, (x1, y1-10), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                            except Exception as e:
                                print(f"개별 객체 처리 중 오류: {str(e)}")
                                continue
            
            return detections, img
        except Exception as e:
            print(f"이미지 처리 중 상세 오류: {str(e)}")
            raise

    def combine_predictions(self, yolo_class, cnn_class, yolo_conf, cnn_conf):
        # YOLO와 CNN의 결과가 일치하는 경우
        if yolo_class == cnn_class:
            return {
                'class': yolo_class,
                'confidence': (yolo_conf + cnn_conf) / 2
            }
        
        # 결과가 불일치하는 경우, 더 높은 신뢰도를 가진 모델의 결과 선택
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
        if isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
        # 이미지 전처리 개선
        img_tensor = self.transform(image).unsqueeze(0)
        
        with torch.no_grad():
            outputs = self.cnn_model(img_tensor)
            probabilities = F.softmax(outputs, dim=1)[0]
            
            # 더 엄격한 분류 기준 적용
            cat_prob = probabilities[0].item()
            dog_prob = probabilities[1].item()
            
            if abs(cat_prob - dog_prob) < 0.2:  # 확실하지 않은 경우
                return {
                    'class': 'unknown',
                    'confidence': max(cat_prob, dog_prob)
                }
            
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

class PetDetectorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("반려동물 탐지 프로그램")
        self.root.geometry("1200x800")
        
        # 반려동물 탐지기 초기화
        self.detector = PetDetector()
        
        self.setup_gui()
        
    def setup_gui(self):
        # 메인 프레임
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 이미지 표시 영역
        self.canvas = tk.Canvas(main_frame, width=800, height=600)
        self.canvas.grid(row=0, column=0, rowspan=6, padx=5, pady=5)
        
        # 컨트롤 프레임
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 버튼들
        ttk.Button(control_frame, text="이미지 선택", command=self.load_image).grid(row=0, column=0, pady=5)
        ttk.Button(control_frame, text="탐지 시작", command=self.detect_pets).grid(row=1, column=0, pady=5)
        ttk.Button(control_frame, text="저장", command=self.save_image).grid(row=2, column=0, pady=5)
        
        # 결과 표시 영역
        self.result_text = tk.Text(control_frame, width=40, height=20)
        self.result_text.grid(row=3, column=0, pady=5)
        
        self.current_image = None
        self.processed_image = None
        
    def load_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff")]
        )
        if file_path:
            self.current_image_path = file_path
            image = Image.open(file_path)
            image.thumbnail((800, 600))
            photo = ImageTk.PhotoImage(image)
            
            self.canvas.config(width=photo.width(), height=photo.height())
            self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
            self.canvas.image = photo
            
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, "이미지가 로드되었습니다.\n")
    
    def detect_pets(self):
        if not hasattr(self, 'current_image_path'):
            messagebox.showerror("오류", "먼저 이미지를 선택해주세요!")
            return
            
        try:
            detections, processed_img = self.detector.process_image(self.current_image_path)
            
            # 결과 표시
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
            print(f"에러 상세 정보: {str(e)}")  # 콘솔에 에러 출력
            messagebox.showerror("오류", f"이미지 처리 중 오류가 발생했습니다:\n{str(e)}")
    
    def save_image(self):
        if self.processed_image is None:
            messagebox.showerror("오류", "저장할 이미지가 없습니다!")
            return
            
        file_path = filedialog.asksaveasfilename(
            defaultextension=".jpg",
            filetypes=[("JPEG files", "*.jpg"), ("PNG files", "*.png")]
        )
        if file_path:
            cv2.imwrite(file_path, self.processed_image)
            messagebox.showinfo("성공", "이미지가 저장되었습니다!")

def main():
    root = tk.Tk()
    app = PetDetectorGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()