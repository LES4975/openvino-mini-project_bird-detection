import gradio as gr
import argparse
from bird_detect.bird_fear import BirdDetector
import os
import cv2

def parse_args():
    parser = argparse.ArgumentParser(description='Bird Detection using YOLO11')
    parser.add_argument(
        '--model_path',
        type=str,
        default="./yolo11s_openvino_model_int8/yolo11s_quant.xml",
        help='Path to the YOLO11 model file (default: ./yolo11s_openvino_model_int8/yolo11s_quant.xml)'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=7860,
        help='Port number for Gradio interface (default: 7860)'
    )
    parser.add_argument(
        '--share',
        action='store_true',
        help='Share the Gradio interface publicly'
    )
    return parser.parse_args()

def detect_bird(image, model_path):
    """
    Gradio interface function for bird detection
    """
    try:
        # 입력 이미지가 없는 경우 처리
        if image is None:
            return None, "Please upload an image first."
        
        # Initialize detector with model path
        detector = BirdDetector(model_path)
        
        # Save uploaded image temporarily
        temp_image_path = "./temp_uploaded_image.jpg"
        image.save(temp_image_path)
        
        # cv2를 사용하여 이미지 읽기 (참고 코드와 동일하게)
        cv_image = cv2.imread(temp_image_path)
        
        # Run detection on the image
        bird_count, output_image = detector.bird_detect(cv_image)
        
        # 탐지 결과 정보 생성
        detection_info = f"🐦 Birds detected: {bird_count}\n"
        detection_info += f"📍 Model used: {os.path.basename(model_path)}\n"
        detection_info += f"✅ Detection completed successfully"
        
        # 임시 파일 정리
        try:
            if os.path.exists(temp_image_path):
                os.remove(temp_image_path)
        except:
            pass
        
        # 결과 반환: (출력 이미지, 탐지 정보)
        return output_image, detection_info
        
    except Exception as e:
        error_msg = f"❌ Error occurred: {str(e)}"
        return None, error_msg

def create_gradio_interface(model_path):
    """
    Create and configure Gradio interface
    """
    # Create the interface
    with gr.Blocks(title="Bird Detection with YOLO11") as demo:
        gr.Markdown("# 🐦 Bird Detection using YOLO11")
        gr.Markdown("Upload an image to detect birds using YOLO11 model")
        
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(
                    type="pil",
                    label="Upload Image",
                    height=400
                )
                model_input = gr.Textbox(
                    value=model_path,
                    label="Model Path",
                    placeholder="Enter model path...",
                    interactive=True
                )
                detect_btn = gr.Button("🔍 Detect Birds", variant="primary")
                
            with gr.Column():
                output = gr.Image(
                    label="Detection Result",
                    height=400
                )
                # Optional: Add text output for detection info
                info_output = gr.Textbox(
                    label="Detection Info",
                    lines=3,
                    interactive=False
                )
        
        # Set up the detection function - outputs를 두 개로 수정
        detect_btn.click(
            fn=detect_bird,
            inputs=[image_input, model_input],
            outputs=[output, info_output]  # 이미지와 정보 모두 출력
        )
        
        # Example images (optional)
        gr.Examples(
            examples=[
                ["./bird_detect/bird_image/IE002877093_STD.jpg"],
            ],
            inputs=image_input,
            label="Example Images"
        )
        
        return demo

def main():
    args = parse_args()
    
    # Create Gradio interface
    demo = create_gradio_interface(args.model_path)
    
    # Launch the interface
    demo.launch(
        server_port=args.port,
        share=args.share,
        inbrowser=True
    )

if __name__ == "__main__":
    main()