import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import tempfile
import os

# 设置页面配置
st.set_page_config(
    page_title="YOLOv12 目标检测系统",
    page_icon="🔍",
    layout="wide"
)

# 数据集类别映射
CLASS_MAP = {
    "Prionailurus bengalensis": "豹猫",
    "Vulpes vulpes": "赤狐",
    "Muntiacus vaginalis": "赤麂",
    "Paguma larvata": "果子狸",
    "Ursus thibetanus": "黑熊",
    "Cervus nippon": "梅花鹿",
    "Macaca mulatta": "猕猴",
    "Lepus sinensis": "野兔",
    "Sus scrofa": "野猪",
    "Naemorhedus griseus": "中华斑羚"
}

# 应用标题
st.title("🎯 YOLOv12 目标检测系统")

# 初始化模型
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# 侧边栏
with st.sidebar:
    st.header("⚙️ 设置")
    
    conf_threshold = st.slider("置信度阈值", 0.0, 1.0, 0.25, 0.05)
    iou_threshold = st.slider("IoU阈值", 0.0, 1.0, 0.45, 0.05)
    
    st.header("🚀 功能")
    detection_mode = st.radio("检测模式", ["图片检测", "视频检测"])

# 主界面
col1, col2 = st.columns(2)

with col1:
    st.subheader("🖼️ 原始图像")
    raw_placeholder = st.empty()

with col2:
    st.subheader("🔍 检测结果")  
    result_placeholder = st.empty()

# 检测结果
st.subheader("📊 检测结果")
table_placeholder = st.empty()

# 图片检测
if detection_mode == "图片检测":
    uploaded_file = st.file_uploader("上传图片", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            image_np = np.array(image)
            raw_placeholder.image(image, use_column_width=True)
            
            with st.spinner("检测中..."):
                results = model.predict(image_np, conf=conf_threshold, iou=iou_threshold)
                result_image = results[0].plot()
                result_image_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
                result_placeholder.image(result_image_rgb, use_column_width=True)
                
                if len(results[0].boxes) > 0:
                    detections = []
                    for det in results[0].boxes:
                        latin_name = results[0].names[int(det.cls)]
                        chinese_name = CLASS_MAP.get(latin_name, latin_name)
                        conf_val = float(det.conf)
                        x = float(det.xyxy[0][0])
                        y = float(det.xyxy[0][1])
                        
                        detections.append({
                            "类别": chinese_name,
                            "置信度": f"{conf_val:.2f}",
                            "位置(x)": f"{x:.2f}",
                            "位置(y)": f"{y:.2f}"
                        })
                    
                    table_placeholder.dataframe(detections)
                else:
                    table_placeholder.warning("未检测到目标")
                    
        except Exception as e:
            st.error(f"错误: {str(e)}")

# 视频检测
elif detection_mode == "视频检测":
    st.info("视频检测功能需要较长时间处理")
    uploaded_video = st.file_uploader("上传视频", type=['mp4', 'avi', 'mov'])
    
    if uploaded_video and st.button("开始检测"):
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_video.read())
            video_path = tmp_file.name
        
        try:
            cap = cv2.VideoCapture(video_path)
            frame_placeholder = st.empty()
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                results = model.predict(frame, conf=conf_threshold, iou=iou_threshold)
                result_frame = results[0].plot()
                result_frame_rgb = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(result_frame_rgb)
            
            cap.release()
            st.success("检测完成!")
            
        except Exception as e:
            st.error(f"错误: {str(e)}")
        finally:
            if os.path.exists(video_path):
                os.unlink(video_path)

st.markdown("---")
st.markdown("YOLOv12 目标检测系统")
