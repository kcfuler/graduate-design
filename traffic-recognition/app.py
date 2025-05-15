import gradio as gr
from PIL import Image
import os
from ultralytics import YOLO, ASSETS  # 确保 ASSETS 也被导入

# --- 新增配置 ---
MODEL_DIR = "models"  # 模型存放目录
loaded_models = {}  # 用于缓存加载的模型
# --- 结束新增配置 ---

# --- 新增函数：获取可用模型列表 ---


def get_available_models():
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)  # 如果目录不存在则创建
    if not os.path.isdir(MODEL_DIR):
        print(f"错误: '{MODEL_DIR}' 不是一个有效的目录。")
        return []

    models = [f for f in os.listdir(
        MODEL_DIR) if f.endswith((".pt", ".engine"))]
    if not models:
        print(f"提示: '{MODEL_DIR}' 目录为空或不包含有效的模型文件 (.pt, .engine)。")
        print("请添加模型文件到该目录。例如，您可以将 'yolov8n.pt' 放入其中。")
    return models
# --- 结束新增函数 ---

# --- 修改 predict_image 函数 ---


def predict_image(model_name, img, conf_threshold, iou_threshold):
    """使用可调节的置信度和IOU阈值的YOLO模型预测图像中的对象。"""
    if not model_name:
        print("错误：没有选择模型。请在下拉菜单中选择一个模型。")
        # 可以返回一个提示用户选择模型的图像或 просто None
        # pil_image = Image.new('RGB', (640, 480), color = 'white')
        # from PIL import ImageDraw
        # draw = ImageDraw.Draw(pil_image)
        # draw.text((10, 10), "请先选择一个模型", fill='black')
        # return pil_image
        return None

    if img is None:
        print("错误：没有上传图片。")
        return None

    model_to_use = None
    if model_name in loaded_models:
        model_to_use = loaded_models[model_name]
    else:
        model_path = os.path.join(MODEL_DIR, model_name)
        if not os.path.exists(model_path):
            print(f"错误：模型文件 {model_path} 未找到。")
            return None
        try:
            print(f"正在加载模型: {model_path}...")
            loaded_models[model_name] = YOLO(model_path)
            model_to_use = loaded_models[model_name]
            print(f"模型 {model_name} 加载成功。")
        except Exception as e:
            print(f"加载模型 {model_name} 失败: {e}")
            return None

    if not model_to_use:
        print(f"错误: 模型 {model_name} 未能成功加载或获取。")
        return None

    results = model_to_use.predict(
        source=img,
        conf=conf_threshold,
        iou=iou_threshold,
        show_labels=True,
        show_conf=True,
        imgsz=640,
    )

    for r in results:
        im_array = r.plot()
        im = Image.fromarray(im_array[..., ::-1])
        return im
    return None
# --- 结束修改 predict_image 函数 ---


# 获取可用模型并设置默认值
available_models = get_available_models()
default_model = available_models[0] if available_models else None

# 更新示例，使其包含模型名称
example_list = []
if default_model:
    # 确保ASSETS中的示例图片路径有效
    bus_image_path = ASSETS / "bus.jpg"
    zidane_image_path = ASSETS / "zidane.jpg"

    # 您可能需要确保这些资源文件确实存在于 ultralytics.ASSETS 定义的路径中
    # 如果 ASSETS 路径下的文件不存在，这里的示例会引发错误
    # 为简单起见，我们假设它们存在，但在实际应用中您可能需要更复杂的检查或提供自己的图片
    example_list.append([default_model, str(bus_image_path), 0.25, 0.45])
    example_list.append([default_model, str(zidane_image_path), 0.25, 0.45])


# 创建 Gradio 界面
iface = gr.Interface(
    fn=predict_image,
    inputs=[
        gr.Dropdown(choices=available_models, label="选择模型",
                    value=default_model, interactive=True),  # 新增模型选择下拉框
        gr.Image(type="pil", label="上传图片"),
        gr.Slider(minimum=0.0, maximum=1.0, value=0.25,
                  label="置信度阈值 (Confidence threshold)"),
        gr.Slider(minimum=0.0, maximum=1.0, value=0.45,
                  label="IOU 阈值 (IoU threshold)"),
    ],
    outputs=gr.Image(type="pil", label="检测结果"),
    title="Ultralytics YOLO 目标检测 Gradio 应用 (支持多模型)",
    description="上传图片进行目标检测。请先从下拉菜单中选择一个模型。",
    examples=example_list if example_list else None  # 如果没有默认模型或示例图片，则不显示示例
)

if __name__ == "__main__":
    if not available_models:
        print(
            f"警告: 在 '{MODEL_DIR}' 目录下未找到任何模型文件。应用将启动，但检测功能可能无法正常工作，直到添加模型并重启应用或刷新页面。")
    iface.launch()
