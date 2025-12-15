# 安全帽检测 FastAPI 接口

## 安装依赖

```bash
pip install -r requirements.txt
```

## 启动API服务

```bash
python main.py
```

## API接口说明

### 1. 安全帽检测接口

- **URL**: `POST /check`
- **功能**: 接收图片文件，进行安全帽检测，返回检测结果
- **参数**:
  - `file`: 图片文件 (multipart/form-data)
- **响应**:
  ```json
  {
    "success": true,
    "detections": [
      {
        "id": 1,
        "type": "helmet",  // 或 "other" (未戴安全帽)
        "confidence": 0.95,
        "coordinates": {
          "x1": 120, "y1": 80, "x2": 180, "y2": 140
        }
      }
    ],
    "statistics": {
      "total": 3,
      "helmet": 2,
      "no_helmet": 1
    },
    "result_image": "data:image/png;base64,..."  // 绘制检测框的图片
  }
  ```

### 2. 根路径

- **URL**: `GET /`
- **功能**:index  首页

## 访问API文档

启动服务后，访问: http://localhost:8000/docs

## 模型文件

确保 `src/helmet_head.pt` 模型文件存在，这是YOLOv10训练好的安全帽检测模型。

## 端口配置

默认端口: 8000
可以在 `main.py` 中修改端口号
