import gradio as gr
from PIL import Image, ImageDraw
import os
from ultralytics import YOLO, ASSETS

# --- 标签映射字典 ---
LABEL_MAP = {
    "i2": "非机动车行驶", "pl100": "限制速度 100 km/h",
    "i4": "机动车行驶", "pl120": "限制速度 120 km/h",
    "i5": "靠右侧道路行驶", "pl20": "限制速度 20 km/h",
    "il100": "最低限速 100 km/h", "pl30": "限制速度 30 km/h",
    "il60": "最低限速 60 km/h", "pl40": "限制速度 40 km/h",
    "il80": "最低限速 80 km/h", "pl5": "限制速度 5 km/h",
    "ip": "人行横道", "pl50": "限制速度 50 km/h",
    "p10": "禁止机动车驶入", "pl60": "限制速度 60 km/h",
    "p11": "禁止鸣喇叭", "pl70": "限制速度 70 km/h",
    "p12": "禁止二轮摩托车驶入", "pl80": "限制速度 80 km/h",
    "p19": "禁止向右转弯", "pm20": "限制质量 20 t",
    "p23": "禁止向左转弯", "pm30": "限制质量 30 t",
    "p26": "禁止载货汽车驶入", "pm55": "限制质量 55 t",
    "p27": "禁止运输危险物品车辆驶入", "pn": "禁止停车",
    "p3": "禁止大型客车驶入", "pne": "禁止驶入",
    "p5": "禁止掉头", "pr40": "解除限制速度",
    "p6": "禁止非机动车进入", "w13": "十字交叉路口",
    "pg": "减速让行", "w32": "施工",
    "ph4": "限制高度 4 m", "w55": "注意儿童",
    "ph4.5": "限制高度 4.5 m", "w57": "注意行人",
    "ph5": "限制高度 5 m", "w59": "注意合流"
}
# --- 结束标签映射字典 ---

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
    detected_labels_text = "未检测到标签。"
    placeholder_img = Image.new('RGB', (640, 480), color='lightgray')
    draw = ImageDraw.Draw(placeholder_img)

    if not model_name:
        print("错误：没有选择模型。请在下拉菜单中选择一个模型。")
        draw.text((10, 10), "请先选择一个模型", fill='black')
        return placeholder_img, "错误：请先选择一个模型。"

    if img is None:
        print("错误：没有上传图片。")
        draw.text((10, 10), "请上传一张图片", fill='black')
        return placeholder_img, "错误：请上传一张图片。"

    model_to_use = None
    if model_name in loaded_models:
        model_to_use = loaded_models[model_name]
    else:
        model_path = os.path.join(MODEL_DIR, model_name)
        if not os.path.exists(model_path):
            error_msg = f"错误：模型文件 {model_path} 未找到。"
            print(error_msg)
            draw.text((10, 10), error_msg, fill='black')
            return placeholder_img, error_msg
        try:
            print(f"正在加载模型: {model_path}...")
            loaded_models[model_name] = YOLO(model_path)
            model_to_use = loaded_models[model_name]
            print(f"模型 {model_name} 加载成功。")
        except Exception as e:
            error_msg = f"加载模型 {model_name} 失败: {e}"
            print(error_msg)
            draw.text((10, 10), error_msg, fill='black')
            return placeholder_img, error_msg

    if not model_to_use:
        error_msg = f"错误: 模型 {model_name} 未能成功加载或获取。"
        print(error_msg)
        draw.text((10, 10), error_msg, fill='black')
        return placeholder_img, error_msg

    try:
        results = model_to_use.predict(
            source=img,
            conf=conf_threshold,
            iou=iou_threshold,
            show_labels=True,  # 模型内部绘图仍会使用其自带的标签
            show_conf=True,
            imgsz=640,
        )

        output_image = None
        detected_classes_ids = set()  # 使用集合确保唯一性

        if results and len(results) > 0:
            for r in results:  # 遍历每个Result对象
                if r.boxes and r.boxes.cls is not None:  # 确保有检测框和类别信息
                    # r.boxes.cls 是一个 tensor，包含每个检测框的类别索引
                    # model_to_use.names 是一个字典 {index: class_name_str}
                    # 例如: {0: 'person', 1: 'car', ...}
                    # 我们需要将模型输出的 class_name_str (如 'p5') 映射到我们的 LABEL_MAP
                    for cls_tensor in r.boxes.cls:
                        class_id_from_model = int(
                            cls_tensor.item())  # 获取类别索引 (int)
                        # model.names 是一个 {int_idx: str_label_from_model_yaml} 的字典
                        # 例如 {0: 'i2', 1: 'p5', ...}
                        # 这个 str_label_from_model_yaml 是我们在训练模型时 data.yaml 中定义的名称
                        if class_id_from_model in model_to_use.names:
                            label_key = model_to_use.names[class_id_from_model]
                            detected_classes_ids.add(
                                label_key)  # 添加模型自身的标签名如 'p5'

                # 获取绘制了检测框的图像
                im_array = r.plot()
                output_image = Image.fromarray(im_array[..., ::-1])

        if not output_image:
            # 如果没有检测结果或无法绘图，返回原始图片或占位符
            print("模型预测未返回有效图像或结果为空")
            draw.text((10, 10), "模型预测无结果或处理失败", fill='black')
            output_image = placeholder_img  # 或者 img (原始上传图片)

        if detected_classes_ids:
            detected_labels_list = []
            for label_key in detected_classes_ids:
                description = LABEL_MAP.get(label_key, f"未知标签: {label_key}")
                detected_labels_list.append(f"- {label_key}: {description}")
            detected_labels_text = "图片中检测到的标签类别：\n" + \
                "\n".join(detected_labels_list)
        else:
            detected_labels_text = "图片中未检测到已知标签。"
            if not output_image:  # 如果连图片都没有，也可能是前面出错了
                output_image = placeholder_img
                draw.text((10, 10), detected_labels_text, fill='black')

        return output_image, detected_labels_text

    except Exception as e:
        error_msg = f"处理图像时发生错误: {e}"
        print(error_msg)
        import traceback
        traceback.print_exc()  # 打印详细的错误堆栈
        draw.text((10, 10), error_msg, fill='black')
        return placeholder_img, error_msg


# 获取可用模型并设置默认值
available_models = get_available_models()
default_model = available_models[0] if available_models else None

# --- 新增标签说明 Markdown ---
labels_markdown = """
## 标签类别说明

| 标签    | 类别名称             | 标签    | 类别名称             |
| :------ | :------------------- | :------ | :------------------- |
| i2      | 非机动车行驶         | pl100   | 限制速度 100 km/h    |
| i4      | 机动车行驶           | pl120   | 限制速度 120 km/h    |
| i5      | 靠右侧道路行驶       | pl20    | 限制速度 20 km/h     |
| il100   | 最低限速 100 km/h    | pl30    | 限制速度 30 km/h     |
| il60    | 最低限速 60 km/h     | pl40    | 限制速度 40 km/h     |
| il80    | 最低限速 80 km/h     | pl5     | 限制速度 5 km/h      |
| ip      | 人行横道             | pl50    | 限制速度 50 km/h     |
| p10     | 禁止机动车驶入       | pl60    | 限制速度 60 km/h     |
| p11     | 禁止鸣喇叭           | pl70    | 限制速度 70 km/h     |
| p12     | 禁止二轮摩托车驶入   | pl80    | 限制速度 80 km/h     |
| p19     | 禁止向右转弯         | pm20    | 限制质量 20 t        |
| p23     | 禁止向左转弯         | pm30    | 限制质量 30 t        |
| p26     | 禁止载货汽车驶入     | pm55    | 限制质量 55 t        |
| p27     | 禁止运输危险物品车辆驶入 | pn      | 禁止停车             |
| p3      | 禁止大型客车驶入     | pne     | 禁止驶入             |
| p5      | 禁止掉头             | pr40    | 解除限制速度         |
| p6      | 禁止非机动车进入     | w13     | 十字交叉路口         |
| pg      | 减速让行             | w32     | 施工                 |
| ph4     | 限制高度 4 m         | w55     | 注意儿童             |
| ph4.5   | 限制高度 4.5 m       | w57     | 注意行人             |
| ph5     | 限制高度 5 m         | w59     | 注意合流             |
"""
# --- 结束新增标签说明 Markdown ---


# 创建 Gradio 界面
iface = gr.Interface(
    fn=predict_image,
    inputs=[
        gr.Dropdown(choices=available_models, label="选择模型",
                    value=default_model, interactive=True),
        gr.Image(type="pil", label="上传图片"),
        gr.Slider(minimum=0.0, maximum=1.0, value=0.25,
                  label="置信度阈值 (Confidence threshold)"),
        gr.Slider(minimum=0.0, maximum=1.0, value=0.45,
                  label="IOU 阈值 (IoU threshold)"),
    ],
    outputs=[
        gr.Image(type="pil", label="检测结果图像"),
        gr.Textbox(label="检测到的标签信息")
    ],
    title="基于深度学习的交通标识识别系统",
    description="上传图片进行目标检测。请先从下拉菜单中选择一个模型。检测到的标签及其含义会显示在图像下方。",
    article=labels_markdown
)

if __name__ == "__main__":
    if not available_models:
        print(
            f"警告: 在 '{MODEL_DIR}' 目录下未找到任何模型文件。应用将启动，但检测功能可能无法正常工作，直到添加模型并重启应用或刷新页面。"
        )
    iface.launch()
