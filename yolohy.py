import torch
from ultralytics import YOLO

def main():
    # 1. 加载模型
    model = YOLO('/data2/kaiyun/code/yolo/yolo-project/runs/classify/anquandai_20260818/phone_cls/weights/best.pt')  
    
    # 2. 禁用训练模式，启用推理优化
    model.model.eval()
    
    # 3. 执行导出
    model.export(
        format='onnx',           # 指定导出格式
        dynamic=False,            # 【关键】必须为 True，以兼容 End2End NMS 算子
        opset=11,                # 推荐 opset 17，兼容主流 ONNX Runtime（16易报错）
        batch=1,
        max_det=100,
        conf=0.30,             # NMS 置信度阈值 (nms=True 时生效, 默认 0.25)
        iou=0.4,               # NMS IoU 阈值 (nms=True 时生效, 默认 0.7)
        simplify=True,           # 自动简化计算图，删除冗余节点提升速度
        imgsz=224,               # 固定输入尺寸，建议与训练时保持一致
        end2end=False,
    )
    print("✅ ONNX 模型导出成功！")

if __name__ == '__main__':
    main()
