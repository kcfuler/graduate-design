import gradio as gr
from PIL import Image

from ultralytics import YOLO

# 加载 YOLO 模型，这里使用 yolov8n.pt 作为示例，您可以根据需要更改
model = YOLO("yolov8n.pt")


def predict_image(img, conf_threshold, iou_threshold):
    """使用可调节的置信度和IOU阈值的YOLO模型预测图像中的对象。"""
    results = model.predict(
        source=img,
        conf=conf_threshold,
        iou=iou_threshold,
        show_labels=True,
        show_conf=True,
        imgsz=640,  # 可以根据需要调整图像大小
    )

    # 处理结果并返回绘制了检测框的图像
    # model.predict 返回一个 Results 列表，即使只有一个输入源
    for r in results:
        im_array = r.plot()  # plot() 方法返回一个 NumPy 数组
        # 将 BGR 图像转换为 RGB
        im = Image.fromarray(im_array[..., ::-1])
        return im
    return None  # 如果没有结果则返回None


# 创建 Gradio 界面
iface = gr.Interface(
    fn=predict_image,
    inputs=[
        gr.Image(type="pil", label="上传图片"),
        gr.Slider(minimum=0.0, maximum=1.0, value=0.25,
                  label="置信度阈值 (Confidence threshold)"),
        gr.Slider(minimum=0.0, maximum=1.0, value=0.45,
                  label="IOU 阈值 (IoU threshold)"),
    ],
    outputs=gr.Image(type="pil", label="检测结果"),
    title="Ultralytics YOLO 目标检测 Gradio 应用",
    description="上传图片进行目标检测。默认使用 YOLOv8n 模型。",
    examples=[
        # 您可以在此处添加示例图片路径，确保路径正确或 ASSETS 已正确定义
        # 例如: ["path/to/your/bus.jpg", 0.25, 0.45],
        # ["path/to/your/zidane.jpg", 0.25, 0.45],
    ],
)

if __name__ == "__main__":
    iface.launch()
