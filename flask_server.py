from flask import Flask, request, jsonify, send_file, render_template
import os
import uuid
from PIL import Image
from ultralytics import YOLO
import sys
from io import BytesIO
import requests
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import io
import cv2  # cv2 모듈 임포트 추가
from flask_cors import CORS
import torch
from ultralytics import Explorer
from ultralytics.data.explorer import plot_query_result
import numpy as np
import matplotlib.pyplot as plt
import random
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
import matplotlib
matplotlib.use('Agg') 

sys.path.append('C:/team/team')
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'team.settings')
django.setup()

from modelservice.models import ImageModel, PredictionResult

app = Flask(__name__)

# CORS 설정 적용
CORS(app)

trained_model = YOLO(r'C:\team\best.pt')  # 'cuda'를 사용하여 GPU 활성화
class_labels = {0: 'pill', 1: 'powder', 2: 'syrup', 3: 'ointment'}

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 이미지 파일을 받아옴
        image_file = request.files['image']
        
        # 이미지 파일을 PIL 이미지로 변환
        img = Image.open(image_file)

        # PIL 이미지를 NumPy 배열로 변환
        img_np = np.array(img)

        # NumPy 배열을 RGB에서 BGR로 변환
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # YOLO 모델을 사용하여 이미지 분석
        results = trained_model(img_np)

        # 분석 결과를 가공  
        predictions = []
        for result in results:
           for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        
                # 클래스 인덱스 가져오기
                class_index = int(box.cls.item())  # 클래스 인덱스를 스칼라로 변환하여 가져옵니다
        
                # 클래스 레이블 가져오기
                class_label = class_labels[class_index]
        
                # 바운딩 박스 그리기
                cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 0, 255), 2)

                # 클래스 이름을 이미지에 출력
                cv2.putText(img_np, class_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
                predictions.append({'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'class': class_label})  # 클래스 정보를 추가

        # 저장할 경로 지정
        save_path = 'C:/team/team/images'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # 유니크한 파일명 생성
        unique_filename = str(uuid.uuid4()) + '.jpg'
        save_file = os.path.join(save_path, unique_filename)

        # 이미지 저장
        cv2.imwrite(save_file, img_np)

        # URL 생성
        url = request.host_url + 'images/' + unique_filename

        # 이미지 모델 생성 및 저장
        image_model = ImageModel(image=unique_filename)
        image_model.save()

        # 예측 결과를 데이터베이스에 저장
        for prediction in predictions:
            prediction_result = PredictionResult(
               image_path=image_model.image,  # 이미지 경로 저장
               x1=prediction['x1'],
               y1=prediction['y1'],
               x2=prediction['x2'],
               y2=prediction['y2'],
               class_label=prediction['class']
               )
            prediction_result.save()
        # URL을 JSON 형태로 반환
        return jsonify({'url': url, 'prediction_result_id': prediction_result.id})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/')
def home():
    return "Welcome to the Flask server!"

if __name__ == '__main__':
    app.run(debug=True)