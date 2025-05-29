import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import argparse
from PIL import Image
from torchvision import transforms
import random

# 导入pytorch_grad_cam相关组件
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# 导入项目相关模块
from vit_train_sa2vp import Dual_model
from datasets import build_dataset


class SA2VPWrapper(torch.nn.Module):
    """专为SA²VP模型设计的包装器，支持GradCAM计算"""
    def __init__(self, model):
        super(SA2VPWrapper, self).__init__()
        self.model = model
        self.vit_base = model.vit_base
        self.class_head = model.class_head
        
        # 保存激活的钩子
        self.activations = None
        # 保存梯度的钩子
        self.gradients = None
        
        # 注册钩子到目标层 (最后一个transformer层)
        target_layer = self._get_target_layer()
        if target_layer is not None:
            target_layer.register_forward_hook(self._activation_hook)
            target_layer.register_full_backward_hook(self._gradient_hook)
    
    def _get_target_layer(self):
        """获取合适的目标层"""
        if hasattr(self.vit_base.transformer.encoder, 'layer'):
            # 找到最后一个transformer层
            return self.vit_base.transformer.encoder.layer[-1]
        return None
    
    def _activation_hook(self, module, input, output):
        """保存前向传播激活值"""
        if isinstance(output, tuple):
            self.activations = output[0].detach()
        else:
            self.activations = output.detach()
    
    def _gradient_hook(self, module, grad_input, grad_output):
        """保存反向传播梯度"""
        if isinstance(grad_output, tuple):
            self.gradients = grad_output[0].detach()
        else:
            self.gradients = grad_output.detach()
    
    def forward(self, x):
        """只返回第一个分类结果，适配GradCAM需求"""
        # 使用原始模型前向传播
        x, _ = self.model(x)
        return x
    
    def get_cam_weights(self):
        """获取CAM权重，模拟GradCAM的实现"""
        if self.gradients is None:
            return None
            
        # 确保梯度和激活维度正确
        if self.gradients.ndim < 3 or self.activations.ndim < 3:
            print(f"梯度形状: {self.gradients.shape}, 激活形状: {self.activations.shape}")
            return None
        
        # 获取梯度在类令牌(CLS token)上的平均值
        pooled_gradients = torch.mean(self.gradients[:, 0, :], dim=0)
        
        # 权重激活
        activations = self.activations
        
        # 激活 * 梯度权重
        for i in range(activations.size(2)):
            activations[:, :, i] *= pooled_gradients[i]
            
        # 在特征维度上平均
        cam = torch.mean(activations, dim=2)
        
        # 去除CLS令牌，保留patch令牌
        cam = cam[:, 1:]
        
        # 重塑为2D特征图
        batch_size = cam.size(0)
        h = w = int(np.sqrt(cam.size(1)))
        cam = cam.reshape(batch_size, h, w)
        
        # 正则化
        with torch.no_grad():
            cam = cam - cam.min()
            cam = cam / (cam.max() + 1e-7)
        
        return cam.cpu().numpy()


def load_model(args, checkpoint_path):
    """载入SA²VP模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = Dual_model(args)
    
    # 加载权重
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    
    return model, device


def get_class_name(dataset_name, class_idx):
    """根据数据集获取类别名称（如果可用）"""
    # 可以根据需要为各个数据集添加类别名称
    class_names = {
        'CUB': None,  # 可以添加鸟类名称
        'CIFAR100': None,  # 可以添加CIFAR100类别名称
        # 其他数据集...
    }
    
    if dataset_name in class_names and class_names[dataset_name] is not None:
        return class_names[dataset_name][class_idx]
    else:
        return f"类别 {class_idx}"


def load_and_preprocess_image(img_path, input_size=224):
    """加载并预处理图像
    
    参数:
        img_path: 图像路径
        input_size: 输入尺寸
    
    返回:
        预处理后的图像张量
    """
    # 定义图像预处理
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        normalize
    ])
    
    # 加载图像
    img = Image.open(img_path).convert('RGB')
    
    # 应用预处理
    img_tensor = transform(img).unsqueeze(0)  # 添加批次维度
    
    return img_tensor


def select_sample_from_dataset(dataset, idx=None, class_id=None):
    """从数据集中选择一个样本
    
    参数:
        dataset: 数据集
        idx: 指定的样本索引，如果为None则随机选择
        class_id: 指定的类别ID，如果不为None则从该类别中随机选择样本
    
    返回:
        样本索引, 样本图像, 样本标签
    """
    if idx is not None:
        # 直接返回指定索引的样本
        sample_idx = idx
        img, label = dataset[sample_idx]
        return sample_idx, img, label
    
    if class_id is not None:
        # 找出指定类别的所有样本
        class_indices = []
        for i in range(len(dataset)):
            _, label = dataset[i]
            if label == class_id:
                class_indices.append(i)
        
        if not class_indices:
            raise ValueError(f"在数据集中找不到类别ID为{class_id}的样本")
        
        # 随机选择一个指定类别的样本
        sample_idx = random.choice(class_indices)
        img, label = dataset[sample_idx]
        return sample_idx, img, label
    
    # 随机选择一个样本
    sample_idx = random.randint(0, len(dataset) - 1)
    img, label = dataset[sample_idx]
    return sample_idx, img, label


def generate_gradcam(model_wrapper, img_tensor, target_class=None, cam_method='gradcam'):
    """生成不同类型的CAM可视化
    
    参数:
        model_wrapper: 模型包装器
        img_tensor: 输入图像张量
        target_class: 目标类别，如果为None则使用预测类别
        cam_method: CAM方法，可选'gradcam'、'gradcam++'、'scorecam'
    
    返回:
        cam: CAM热力图
        pred_class: 预测类别
        confidence: 预测置信度
    """
    device = next(model_wrapper.parameters()).device
    img_tensor = img_tensor.to(device)
    
    # 获取模型预测
    with torch.no_grad():
        outputs = model_wrapper(img_tensor)
        pred_class = torch.argmax(outputs, dim=1).item()
        probs = torch.nn.functional.softmax(outputs, dim=1)
        confidence = probs[0, pred_class].item()
    
    # 如果没有指定目标类别，使用预测的类别
    if target_class is None:
        target_class = pred_class
    
    # 使用自定义方法生成CAM
    model_wrapper.activations = None
    model_wrapper.gradients = None
    
    # 进行前向传播和反向传播
    model_wrapper.eval()
    outputs = model_wrapper(img_tensor)
    
    model_wrapper.zero_grad()
    one_hot = torch.zeros_like(outputs)
    one_hot[0, target_class] = 1
    outputs.backward(gradient=one_hot, retain_graph=True)
    
    # 获取CAM权重
    cam = model_wrapper.get_cam_weights()
    
    # 如果自定义方法失败，使用标准CAM库
    if cam is None:
        print(f"使用标准{cam_method}库...")
        
        # 获取模型的最后一层
        if hasattr(model_wrapper.vit_base.transformer.encoder, 'layer'):
            target_layers = [model_wrapper.vit_base.transformer.encoder.layer[-1]]
        else:
            target_layers = [model_wrapper.vit_base.transformer.encoder]
        
        # 根据指定方法创建CAM实例
        if cam_method == 'gradcam++':
            cam_instance = GradCAMPlusPlus(
                model=model_wrapper,
                target_layers=target_layers,
                use_cuda=torch.cuda.is_available()
            )
        elif cam_method == 'scorecam':
            cam_instance = ScoreCAM(
                model=model_wrapper,
                target_layers=target_layers,
                use_cuda=torch.cuda.is_available()
            )
        else:  # 默认使用GradCAM
            cam_instance = GradCAM(
                model=model_wrapper,
                target_layers=target_layers,
                use_cuda=torch.cuda.is_available()
            )
        
        # 计算CAM
        targets = [ClassifierOutputTarget(target_class)]
        cam = cam_instance(input_tensor=img_tensor, targets=targets)
        cam = cam[0, :]
    else:
        cam = cam[0]
    
    return cam, pred_class, confidence


def denormalize_image(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """反归一化张量图像到原始RGB空间"""
    tensor = tensor.clone().detach().cpu()
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    tensor = tensor.clamp(0, 1)
    return tensor


def visualize_cam(img_tensor, cam, pred_class, confidence, true_class, output_path, class_name=None):
    """可视化CAM结果
    
    参数:
        img_tensor: 输入图像张量
        cam: CAM热力图
        pred_class: 预测类别
        confidence: 预测置信度
        true_class: 真实类别（可以为None，表示用户自定义图像）
        output_path: 输出路径
        class_name: 类别名称
    """
    # 反归一化图像张量
    img_denorm = denormalize_image(img_tensor[0])
    img_np = img_denorm.permute(1, 2, 0).numpy()
    
    # 调整CAM大小以匹配图像
    cam_resized = cv2.resize(cam, (img_np.shape[1], img_np.shape[0]))
    
    # 创建热力图
    cam_heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    cam_heatmap = cam_heatmap / 255.0
    cam_heatmap = np.float32(cam_heatmap)
    
    # 叠加热力图和原始图像
    alpha = 0.5
    cam_result = (1-alpha) * img_np + alpha * cam_heatmap
    cam_result = np.clip(cam_result, 0, 1)
    
    # 创建高质量可视化
    plt.figure(figsize=(16, 5))
    
    # 原始图像
    plt.subplot(1, 3, 1)
    plt.imshow(img_np)
    if true_class is not None:
        plt.title(f'原始图像 (真实类别: {true_class})', fontsize=12)
    else:
        plt.title('原始图像', fontsize=12)
    plt.axis('off')
    
    # 热力图
    plt.subplot(1, 3, 2)
    plt.imshow(cam_resized, cmap='jet')
    plt.title('类激活热力图', fontsize=12)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis('off')
    
    # 叠加结果
    class_info = class_name if class_name else f"类别: {pred_class}"
    plt.subplot(1, 3, 3)
    plt.imshow(cam_result)
    plt.title(f'{class_info}, 置信度: {confidence:.2f}', fontsize=12)
    plt.axis('off')
    
    # 保存结果
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"已保存CAM可视化结果到 {output_path}")
    
    return cam_result


def main():
    parser = argparse.ArgumentParser(description='SA²VP模型的Grad-CAM可视化')
    
    # 数据集和模型参数
    parser.add_argument('--data_set', type=str, default='CUB', help='数据集名称')
    parser.add_argument('--nb_classes', type=int, default=200, help='类别数量')
    parser.add_argument('--checkpoint', type=str, required=True, help='模型检查点路径')
    parser.add_argument('--output_dir', type=str, default='gradcam_results', help='结果保存目录')
    
    # 可视化选项
    parser.add_argument('--cam_method', type=str, default='gradcam', choices=['gradcam', 'gradcam++', 'scorecam'], 
                        help='CAM方法选择')
    parser.add_argument('--target_class', type=int, default=None, help='目标类别ID（不指定则使用预测类别）')
    
    # 样本选择选项
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--image_path', type=str, default=None, help='要可视化的自定义图像路径')
    group.add_argument('--sample_idx', type=int, default=None, help='要可视化的数据集样本索引')
    group.add_argument('--class_id', type=int, default=None, help='要可视化的类别ID，将随机选择该类别的一个样本')
    group.add_argument('--random_samples', type=int, default=1, help='随机选择的样本数量')
    
    # 数据预处理参数
    parser.add_argument('--imagenet_default_mean_and_std', action='store_true', default=True)
    parser.add_argument('--input_size', type=int, default=224, help='输入图像大小')
    parser.add_argument('--my_mode', type=str, default='trainval_test', help='数据集模式')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载模型
    model, device = load_model(args, args.checkpoint)
    print("模型加载完成...")
    
    # 创建模型包装器
    model_wrapper = SA2VPWrapper(model)
    
    # 处理自定义图像
    if args.image_path is not None:
        # 加载并预处理图像
        print(f"加载自定义图像: {args.image_path}")
        img_tensor = load_and_preprocess_image(args.image_path, args.input_size)
        
        # 生成CAM可视化
        cam, pred_class, confidence = generate_gradcam(
            model_wrapper, img_tensor, args.target_class, args.cam_method
        )
        
        # 获取类别名称
        class_name = get_class_name(args.data_set, pred_class)
        
        # 从图像路径中提取文件名
        file_name = os.path.splitext(os.path.basename(args.image_path))[0]
        output_path = os.path.join(args.output_dir, f"{file_name}_cam.png")
        
        # 可视化并保存结果（自定义图像没有真实标签）
        visualize_cam(img_tensor, cam, pred_class, confidence, None, output_path, class_name)
        
        print(f"图像预测结果 - 类别: {pred_class}, 置信度: {confidence:.4f}")
    else:
        # 加载验证数据集
        dataset_val, _ = build_dataset(is_train=False, args=args)
        print(f"验证数据集大小: {len(dataset_val)}")
        
        # 处理多个样本的情况
        if args.random_samples > 1:
            for i in range(args.random_samples):
                sample_idx, img, label = select_sample_from_dataset(dataset_val, class_id=args.class_id)
                
                # 生成CAM可视化
                cam, pred_class, confidence = generate_gradcam(
                    model_wrapper, img.unsqueeze(0), args.target_class, args.cam_method
                )
                
                # 获取类别名称
                class_name = get_class_name(args.data_set, pred_class)
                
                # 可视化并保存结果
                output_path = os.path.join(args.output_dir, f"sample_{sample_idx}_cam.png")
                visualize_cam(img.unsqueeze(0), cam, pred_class, confidence, label, output_path, class_name)
                
                print(f"样本 {sample_idx} - 真实类别: {label}, 预测类别: {pred_class}, 置信度: {confidence:.4f}")
        else:
            # 选择单个样本
            sample_idx, img, label = select_sample_from_dataset(
                dataset_val, idx=args.sample_idx, class_id=args.class_id
            )
            
            # 生成CAM可视化
            cam, pred_class, confidence = generate_gradcam(
                model_wrapper, img.unsqueeze(0), args.target_class, args.cam_method
            )
            
            # 获取类别名称
            class_name = get_class_name(args.data_set, pred_class)
            
            # 可视化并保存结果
            output_path = os.path.join(args.output_dir, f"sample_{sample_idx}_cam.png")
            visualize_cam(img.unsqueeze(0), cam, pred_class, confidence, label, output_path, class_name)
            
            print(f"样本 {sample_idx} - 真实类别: {label}, 预测类别: {pred_class}, 置信度: {confidence:.4f}")


if __name__ == '__main__':
    main() 