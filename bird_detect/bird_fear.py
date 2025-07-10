from ultralytics import YOLO
from openvino.runtime import Core
import numpy as np
import cv2
from pathlib import Path



class BirdDetector : 
    def __init__(self, model_path:str):

        
        int8_model_det_path = Path(model_path)
        self.model = YOLO(int8_model_det_path.parent, task='detect')
        # det_model = YOLO(int8_model_det_path.parent, task="detect")
        self.bird_class_id = 14  # COCO 데이터셋에서 새(bird)의 클래스 ID


    def bird_detect(self, image) :
        """ 새 발견 기능 구현 """
        """
        이미지에서 새 탐지
        
        Args:
            image: 입력 이미지 (numpy array 또는 파일 경로)
            
        Returns:
            tuple: (탐지된 새의 개수, 탐지 결과가 그려진 이미지)
        """
        print(type(image))
        results = self.model(image)
        bird_count = 0
        
        # 원본 이미지 복사 (탐지 결과를 그리기 위해)
        result_image = image.copy()
        print(type(result_image))
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    
                    # 바운딩 박스 좌표 추출
                    xyxy = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = xyxy.astype(int)
                    
                    # 클래스 이름 가져오기
                    class_name = self.model.names[class_id]
                    
                    # 새만 따로 카운트
                    if class_id == self.bird_class_id:
                        bird_count += 1
                        # 새는 빨간색으로 표시
                        color = (0, 0, 255)  # 빨간색 (BGR)
                        thickness = 3
                    else:
                        color = (0, 255, 0) 
                        thickness = 3
            
                    # 바운딩 박스 그리기
                    cv2.rectangle(result_image, (x1, y1), (x2, y2), color, thickness)
                    
                    # 라벨 텍스트 준비
                    label = f"{class_name}: {confidence:.2f}"
                    
                    # 텍스트 배경 사각형 그리기
                    (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(result_image, (x1, y1 - text_height - 5), (x1 + text_width, y1), color, -1)
                    
                    # 텍스트 그리기
                    cv2.putText(result_image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return bird_count, result_image


    def create_notice_image(self, bird_detected=False):
        """경고 이미지 생성"""
        img_notice_path = '/home/paper/workspace/yolo11/openvino-mini-project_bird-detection/bird_detect/notice/notice.png'
        
        # 경고 이미지 로드
        try:
            notice_img = cv2.imread(img_notice_path)
            if notice_img is None:
                # 경고 이미지가 없으면 빈 이미지 생성
                notice_img = np.ones((300, 400, 3), dtype=np.uint8) * 255
                # 경고 텍스트 추가
                cv2.putText(notice_img, "BIRD ALERT!", (50, 150), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
        except:
            # 기본 경고 이미지 생성
            notice_img = np.ones((300, 400, 3), dtype=np.uint8) * 255
            cv2.putText(notice_img, "BIRD ALERT!", (50, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
        
        # 새가 감지되면 빨간색 테두리 추가
        if bird_detected:
            h, w = notice_img.shape[:2]
            border_thickness = 10
            # 빨간색 테두리 그리기
            cv2.rectangle(notice_img, (0, 0), (w-1, h-1), (0, 0, 255), border_thickness)
            
        return notice_img
    
    def detect_start(self, image_path:str):

        image = cv2.imread(image_path)


        # 새 탐지
        bird_count, output_image = self.bird_detect(image)
        
        # 새가 있는지 확인
        bird_detected = bird_count > 0
        
        # 경고 이미지 생성
        notice_img = self.create_notice_image(bird_detected)
        
        # 결과 출력
        if bird_detected:
            print(f"새 {bird_count}마리 발견!")
        else:
            print("새 없음")
        
        # # 이미지 보여주기
        # cv2.imshow('Notice', notice_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()


        # 이미지 보여주기
        # cv2.imshow('Original Image', image)           # 원본 이미지
        cv2.imshow('Detection Result', output_image)  # 탐지 결과 이미지
        cv2.imshow('Notice', notice_img)              # 경고 이미지
        cv2.waitKey(0)
        cv2.destroyAllWindows()

