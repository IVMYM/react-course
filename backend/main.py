
from fastapi import FastAPI, File, UploadFile, HTTPException
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
import cv2
from fastapi.staticfiles import StaticFiles
import numpy as np
from ultralytics import YOLO
from PIL import Image
import base64
import os

# 初始化FastAPI应用
app = FastAPI(title="安全帽检测API", version="1.0.0")
# 挂载静态文件目录（前端CSS/JS等）
app.mount("/static", StaticFiles(directory="static", html=True), name="static")
    
# 根路由返回index.html
@app.get("/")
async def read_index():
       # 读取 templates 目录下的 index.html 并返回
    print("🚀index.html...")
    index_path = Path("templates/index.html")
    if not index_path.exists():
        return {"error": "index.html not found"}  # 调试用，确认文件是否存在
    with open(index_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())
# 配置CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生产环境中应该限制为具体的域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 加载模型
model_path = os.path.join(os.path.dirname(__file__), 'helmet_head.pt')
model = YOLO(model_path)

# 类别映射
class_names = {0: 'helmet', 1: 'head', 2: 'reflective_clothes', 3: 'other_clothes'}
target_classes = [0, 1]  # 仅处理helmet(0)和head(1)

# 颜色配置
color_helmet = (0, 255, 0)  # 绿色
color_other = (0, 0, 255)   # 红色

@app.post("/api/detect")
async def check_helmet(file: UploadFile = File(...)):
    """
    安全帽检测接口
    接收图片文件，返回检测结果
    """
    try:
        # 验证文件类型
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="请上传图片文件")
        
        # 读取图片内容
        contents = await file.read()
        
        # 转换为OpenCV格式
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="无法读取图片")
        
        # 推理预测
        results = model(img)
        
        detections = []
        result_img = img.copy()
        
        # 处理检测结果
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                
                # 只处理helmet和head
                if cls_id in target_classes:
                    # 坐标转换
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # 判断类别
                    if cls_id == 1:  # head类别
                        label = "other"
                        draw_color = color_other
                    else:  # helmet类别
                        label = class_names[cls_id]
                        draw_color = color_helmet
                    
                    # 添加到检测结果
                    detection = {
                        "id": len(detections) + 1,
                        "type": label,
                        "confidence": conf,
                        "coordinates": {
                            "x1": x1, "y1": y1,
                            "x2": x2, "y2": y2
                        }
                    }
                    detections.append(detection)
                    
                    # 绘制矩形框
                    cv2.rectangle(result_img, (x1, y1), (x2, y2), draw_color, 2)
                    
                    # 绘制标签
                    label_text = f"{label} {conf:.2f}"
                    label_size, _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    label_x = x1
                    label_y = y1 - 10 if y1 - 10 > 10 else y1 + 20
                    
                    # 标签背景框
                    cv2.rectangle(result_img, 
                                 (label_x, label_y - label_size[1] - 5),
                                 (label_x + label_size[0] + 5, label_y + 5),
                                 draw_color, -1)
                    
                    # 标签文字
                    cv2.putText(result_img, label_text, (label_x + 2, label_y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # 将结果图片转换为base64
        _, buffer = cv2.imencode('.png', result_img)
        result_image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # 统计信息
        helmet_count = len([d for d in detections if d["type"] == "helmet"])
        other_count = len([d for d in detections if d["type"] == "other"])
        
        return JSONResponse(content={
            "success": True,
            "detections": detections,
            "statistics": {
                "total": len(detections),
                "helmet": helmet_count,
                "no_helmet": other_count
            },
            "result_image": f"data:image/png;base64,{result_image_base64}"
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"检测失败: {str(e)}")

@app.get("/health")
async def root():
    """API根路径"""
    return {"message": "安全帽检测API服务运行中", "version": "1.0.0"}

if __name__ == "__main__":
    import uvicorn
    print("🚀 启动安全帽检测API服务...")
    print(f"📁 模型路径: {model_path}")
    print("🌐 服务地址: http://localhost:8000")
    print("📋 API文档: http://localhost:8000/docs")
    print("⏹️  按 Ctrl+C 停止服务")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)