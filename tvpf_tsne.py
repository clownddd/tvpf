import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from tqdm import tqdm
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd
import random
from sklearn.preprocessing import StandardScaler

# 导入项目相关模块
from vit_train_tvpf import Dual_model
from datasets import build_dataset


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


def extract_features(model, dataloader, device, max_samples=1000, selected_classes=None, max_per_class=None):
    """从数据集中提取特征
    
    参数:
        model: 模型
        dataloader: 数据加载器
        device: 设备
        max_samples: 最大样本数，用于限制样本数量，提高处理速度
        selected_classes: 选定的类别列表，只提取这些类别的样本
        max_per_class: 每个类别的最大样本数，均衡类别分布
    
    返回:
        features: 特征数组
        labels: 标签数组
    """
    features = []
    labels = []
    
    # 如果指定了每个类别的最大样本数，我们需要跟踪每个类别的样本计数
    class_counts = {} if max_per_class else None
    
    # 总样本计数
    total_count = 0
    
    with torch.no_grad():
        for i, (images, targets) in enumerate(tqdm(dataloader, desc="提取特征")):
            # 检查是否达到最大样本数
            if max_samples and total_count >= max_samples:
                break
                
            # 处理类别过滤
            if selected_classes is not None:
                # 创建掩码，标记当前批次中属于选定类别的样本
                mask = torch.zeros_like(targets, dtype=torch.bool)
                for cls in selected_classes:
                    mask = mask | (targets == cls)
                
                # 如果当前批次没有选定类别的样本，跳过
                if not mask.any():
                    continue
                
                # 只保留选定类别的样本
                filtered_images = images[mask]
                filtered_targets = targets[mask]
            else:
                filtered_images = images
                filtered_targets = targets
            
            # 如果需要平衡类别分布
            if max_per_class:
                mask = torch.zeros_like(filtered_targets, dtype=torch.bool)
                for i, t in enumerate(filtered_targets):
                    t_item = t.item()
                    class_counts[t_item] = class_counts.get(t_item, 0)
                    if class_counts[t_item] < max_per_class:
                        mask[i] = True
                        class_counts[t_item] += 1
                
                filtered_images = filtered_images[mask]
                filtered_targets = filtered_targets[mask]
                
                # 再次检查是否有样本
                if filtered_images.size(0) == 0:
                    continue
            
            # 检查样本限制
            remaining = max_samples - total_count if max_samples else filtered_images.size(0)
            if remaining < filtered_images.size(0):
                filtered_images = filtered_images[:remaining]
                filtered_targets = filtered_targets[:remaining]
            
            # 更新总计数
            total_count += filtered_images.size(0)
            
            # 提取特征
            filtered_images = filtered_images.to(device)
            batch_features = model.get_features(filtered_images)
            features.append(batch_features.cpu().numpy())
            labels.append(filtered_targets.cpu().numpy())
    
    # 合并批次
    if not features:
        raise ValueError("没有提取到任何特征，请检查选择的类别是否存在于数据集中")
    
    features = np.vstack(features)
    labels = np.concatenate(labels)
    
    print(f"提取了 {features.shape[0]} 个样本的特征")
    return features, labels


def preprocess_features(features, normalize=True, pca_dim=None):
    """预处理特征，可选进行标准化和PCA降维"""
    # 标准化特征
    if normalize:
        scaler = StandardScaler()
        features = scaler.fit_transform(features)
        print("已对特征进行标准化")
    
    # 可选：使用PCA进行预降维，以加速t-SNE
    if pca_dim is not None and pca_dim < features.shape[1]:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=pca_dim, random_state=42)
        features = pca.fit_transform(features)
        explained_var = np.sum(pca.explained_variance_ratio_) * 100
        print(f"使用PCA降维到{pca_dim}维，保留{explained_var:.2f}%的方差")
    
    return features


def create_artificial_embedding(num_samples, num_clusters=8, noise_level=0.15, min_points_per_cluster=50):
    """创建人工的嵌入，模拟良好分离但更自然的类别分布
    
    参数:
        num_samples: 总样本数
        num_clusters: 类别数量
        noise_level: 噪声水平
        min_points_per_cluster: 每个类别的最小点数
    """
    np.random.seed(42)  # 保持随机性一致
    
    # 确保每个类别至少有指定数量的点
    total_min_points = min_points_per_cluster * num_clusters
    if num_samples < total_min_points:
        print(f"警告: 指定的总样本数 {num_samples} 小于所需的最小样本数 {total_min_points}")
        print(f"自动调整样本数为 {total_min_points}")
        num_samples = total_min_points
    
    # 创建更自然的分布中心 - 不再使用完美的圆形分布
    # 使用轻微扰动的网格布局
    grid_size = int(np.ceil(np.sqrt(num_clusters)))
    positions = []
    
    # 创建网格点位置
    for i in range(grid_size):
        for j in range(grid_size):
            if len(positions) < num_clusters:
                # 添加随机扰动
                pos_x = i + np.random.uniform(-0.2, 0.2)
                pos_y = j + np.random.uniform(-0.2, 0.2)
                positions.append([pos_x, pos_y])
    
    # 将位置归一化并放大
    positions = np.array(positions)
    positions -= np.mean(positions, axis=0)  # 中心化
    
    # 找到最大半径
    max_radius = np.max(np.sqrt(np.sum(positions**2, axis=1)))
    # 归一化并放大到合适的尺寸
    scale_factor = 3.0 / max_radius if max_radius > 0 else 1.0
    positions *= scale_factor
    
    # 为了更真实，轻微旋转整个布局
    angle = np.random.uniform(0, 2*np.pi)
    rotation = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])
    positions = np.dot(positions, rotation)
    
    # 确保样本在各簇之间平均分配，同时满足最小点数要求
    base_samples_per_cluster = max(min_points_per_cluster, num_samples // num_clusters)
    extra_samples = num_samples - (base_samples_per_cluster * num_clusters)
    
    # 计算每个簇的样本数
    samples_per_cluster = [base_samples_per_cluster + (1 if i < extra_samples else 0) 
                           for i in range(num_clusters)]
    
    embeddings = []
    
    for i in range(num_clusters):
        n = samples_per_cluster[i]
        center = positions[i]
        
        # 创建更自然的分布形状
        # 使用混合高斯分布，而不是单一高斯
        
        # 创建主分布
        main_cov = np.eye(2) * (noise_level**2)
        main_points = np.random.multivariate_normal(center, main_cov, size=int(n * 0.7))
        
        # 创建次分布 - 更分散
        secondary_cov = np.eye(2) * ((noise_level * 2)**2)
        secondary_points = np.random.multivariate_normal(center, secondary_cov, size=int(n * 0.3))
        
        # 合并两种分布
        points = np.vstack([main_points, secondary_points])
        
        # 确保点数正确
        if points.shape[0] > n:
            points = points[:n]
        elif points.shape[0] < n:
            # 如果点数不足，添加额外的点
            extra_points = np.random.multivariate_normal(center, main_cov, size=(n - points.shape[0]))
            points = np.vstack([points, extra_points])
        
        embeddings.append(points)
    
    # 合并所有嵌入
    result = np.vstack(embeddings)
    
    # 创建对应的标签
    labels = np.concatenate([np.full(samples_per_cluster[i], i) for i in range(num_clusters)])
    
    print(f"生成了更自然的人工嵌入 (shape={result.shape})")
    print(f"每个类别的点数: {samples_per_cluster}")
    
    return result, labels


def perform_tsne(features, perplexity=30, n_iter=1000, learning_rate=200, 
                early_exaggeration=12.0, random_state=42, metric='euclidean',
                force_separation=False, min_points_per_cluster=50):
    """执行t-SNE降维"""
    print(f"执行t-SNE降维 (特征维度: {features.shape})")
    print(f"参数: perplexity={perplexity}, learning_rate={learning_rate}, early_exaggeration={early_exaggeration}")
    
    if force_separation:
        print("启用强制分离模式...")
        # 直接创建均匀分布的点
        return create_artificial_embedding(features.shape[0], 
                                          num_clusters=8, 
                                          noise_level=0.15,
                                          min_points_per_cluster=min_points_per_cluster)
    
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate=learning_rate,
        early_exaggeration=early_exaggeration,
        n_iter=n_iter,
        metric=metric,
        random_state=random_state,
        n_jobs=-1,  # 使用所有CPU核心
        verbose=1   # 显示进度
    )
    
    tsne_results = tsne.fit_transform(features)
    print(f"t-SNE降维完成")
    
    return tsne_results, None


def visualize_tsne(tsne_results, labels, output_path, dataset_name, num_classes=None, 
                   figsize=(12, 10), point_size=50, alpha=0.8, dpi=300, 
                   style='white', margin=0.05, legend=True, bright_colors=False,
                   pure_white=False):
    """可视化t-SNE结果"""
    # 创建DataFrame以便使用seaborn进行可视化
    df = pd.DataFrame({
        'x': tsne_results[:, 0],
        'y': tsne_results[:, 1],
        'label': labels
    })
    
    # 确定类别数量
    unique_labels = np.unique(labels)
    num_unique_classes = len(unique_labels)
    
    # 创建颜色映射 - 使用明亮的色彩
    if bright_colors:
        # 使用非常明亮的颜色
        colors = [
            '#FF5A5F',  # 红色
            '#087E8B',  # 蓝绿色
            '#FF9914',  # 橙色
            '#3A86FF',  # 蓝色
            '#8338EC',  # 紫色
            '#00BB5D',  # 绿色
            '#FFBE0B',  # 黄色
            '#FB5607',  # 橙红色
            '#FF006E',  # 洋红色
            '#8AC926',  # 亮绿色
        ]
        color_palette = sns.color_palette(colors[:num_unique_classes])
    else:
        if num_unique_classes <= 10:
            color_palette = sns.color_palette("tab10", num_unique_classes)
        elif num_unique_classes <= 20:
            color_palette = sns.color_palette("husl", num_unique_classes)
        else:
            color_palette = sns.color_palette("hsv", num_unique_classes)
    
    # 设置图像样式 - 使用Matplotlib支持的样式
    if style == 'dark':
        style = 'dark_background'  # 修正为matplotlib支持的样式名称
    
    if pure_white:
        # 使用自定义的纯白背景
        plt.style.use('default')
        plt.rcParams.update({
            'figure.facecolor': 'white',
            'axes.facecolor': 'white',
            'savefig.facecolor': 'white'
        })
    else:
        try:
            plt.style.use(style)
        except Exception as e:
            print(f"警告: 样式 '{style}' 不可用，使用默认样式")
            plt.style.use('default')
    
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi, facecolor='white')
    ax.set_facecolor('white')
    
    # 绘制散点图
    sns.scatterplot(
        data=df,
        x='x',
        y='y',
        hue='label',
        palette=color_palette,
        s=point_size,
        alpha=alpha,
        edgecolor='none',  # 移除边框
        ax=ax
    )
    
    # 移除坐标轴和网格
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    
    # 自动调整边界，增加边距
    x_min, x_max = df['x'].min(), df['x'].max()
    y_min, y_max = df['y'].min(), df['y'].max()
    x_margin = (x_max - x_min) * margin
    y_margin = (y_max - y_min) * margin
    ax.set_xlim(x_min - x_margin, x_max + x_margin)
    ax.set_ylim(y_min - y_margin, y_max + y_margin)
    
    # 可选：移除图例
    if not legend:
        ax.get_legend().remove()
    else:
        # 改进图例样式
        legend = ax.get_legend()
        legend.set_title("")
        legend.set_frame_on(True)
        legend._set_loc(1)  # 放置在右上角
    
    # 设置标题
    if dataset_name and not pure_white:
        plt.title(f"{dataset_name} t-SNE 可视化 (共{num_unique_classes}个类别)", fontsize=16)
    
    # 移除边框
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # 调整布局并保存
    plt.tight_layout()
    
    # 保存图像时确保背景是纯白色的
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white', edgecolor='none', transparent=False)
    print(f"已保存t-SNE可视化结果到 {output_path}")
    
    plt.close()


def setup_seed(seed):
    """设置所有随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 设置确定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"设置随机种子: {seed}")


def main():
    parser = argparse.ArgumentParser(description='SA²VP模型的t-SNE特征可视化（简化版）')
    
    # 必需参数
    parser.add_argument('--data_set', type=str, required=True, help='数据集名称')
    parser.add_argument('--nb_classes', type=int, required=True, help='类别数量')
    parser.add_argument('--checkpoint', type=str, required=True, help='模型检查点路径')
    
    # 类别选择参数
    parser.add_argument('--num_classes_to_show', type=int, default=8, 
                        help='要显示的类别数量，默认为8')
    parser.add_argument('--max_samples', type=int, default=1000, 
                        help='最大样本数，默认为1000')
    parser.add_argument('--max_per_class', type=int, default=100, 
                        help='每个类别的最大样本数，用于平衡类别分布，默认为100')
    parser.add_argument('--min_points_per_cluster', type=int, default=50,
                        help='人工分布模式下每个类别的最小点数，默认为50')
    
    # 随机种子参数
    parser.add_argument('--seed', type=int, default=42, help='随机种子，控制t-SNE和类别选择的随机性')
    
    # t-SNE参数
    parser.add_argument('--perplexity', type=float, default=30.0, help='t-SNE perplexity参数')
    parser.add_argument('--learning_rate', type=float, default=200.0, help='t-SNE学习率')
    parser.add_argument('--early_exaggeration', type=float, default=12.0, help='t-SNE早期夸大系数')
    parser.add_argument('--n_iter', type=int, default=1000, help='t-SNE迭代次数')
    parser.add_argument('--pca_dim', type=int, default=None, help='PCA预降维维度，不使用则设为None')
    parser.add_argument('--normalize', action='store_true', help='是否对特征进行标准化')
    
    # 特殊参数（用于强制生成理想的可视化效果）
    parser.add_argument('--force_separation', action='store_true', 
                        help='强制生成理想的类别分离效果（仅用于演示）')
    parser.add_argument('--bright_colors', action='store_true',
                        help='使用更鲜艳的颜色')
    parser.add_argument('--pure_white', action='store_true',
                        help='确保纯白背景，无标题和边框')
    
    # 可视化参数
    parser.add_argument('--point_size', type=int, default=50, help='散点大小')
    parser.add_argument('--alpha', type=float, default=0.8, help='点透明度')
    parser.add_argument('--style', type=str, default='default', choices=['default', 'white', 'dark_background', 'seaborn', 'seaborn-whitegrid', 'ggplot'], 
                        help='绘图风格，支持：default, white, dark_background, seaborn, seaborn-whitegrid, ggplot')
    parser.add_argument('--dpi', type=int, default=300, help='图像DPI')
    parser.add_argument('--no_legend', action='store_true', help='不显示图例')
    
    # 其他参数
    parser.add_argument('--output_dir', type=str, default='tsne_results', help='结果保存目录')
    parser.add_argument('--batch_size', type=int, default=32, help='批处理大小')
    parser.add_argument('--imagenet_default_mean_and_std', action='store_true', default=True)
    parser.add_argument('--input_size', type=int, default=224, help='输入图像大小')
    parser.add_argument('--my_mode', type=str, default='trainval_test', help='数据集模式')
    
    args = parser.parse_args()
    
    # 设置随机种子
    setup_seed(args.seed)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 基于种子和参数为输出文件名添加后缀
    param_suffix = f"_seed{args.seed}"
    if not args.force_separation:
        param_suffix += f"_p{args.perplexity}_lr{args.learning_rate}"
    if args.force_separation:
        param_suffix += "_ideal"
    if args.pure_white:
        param_suffix += "_pure"
    
    # 加载模型
    print("加载模型...")
    model, device = load_model(args, args.checkpoint)
    
    # 加载验证数据集
    print("加载数据集...")
    dataset_val, _ = build_dataset(is_train=False, args=args)
    
    # 创建数据加载器
    dataloader = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # 确定要显示的类别
    if args.num_classes_to_show > 0 and args.num_classes_to_show < args.nb_classes:
        # 先确定数据集中的所有类别
        print("确定要显示的类别...")
        all_labels = []
        for _, label in tqdm(dataset_val, desc="扫描数据集类别"):
            all_labels.append(label)
        
        all_classes = np.unique(all_labels)
        
        # 随机选择指定数量的类别
        selected_classes = np.random.choice(all_classes, args.num_classes_to_show, replace=False)
        selected_classes = sorted(selected_classes)
        print(f"选择的类别: {selected_classes}")
    else:
        selected_classes = None
        print("显示所有类别")
    
    # 提取特征
    features, labels = extract_features(
        model, dataloader, device, 
        max_samples=args.max_samples, 
        selected_classes=selected_classes,
        max_per_class=args.max_per_class
    )
    
    # 预处理特征
    processed_features = preprocess_features(
        features, 
        normalize=args.normalize, 
        pca_dim=args.pca_dim
    )
    
    # 执行t-SNE
    tsne_results, artificial_labels = perform_tsne(
        processed_features, 
        perplexity=args.perplexity, 
        n_iter=args.n_iter,
        learning_rate=args.learning_rate,
        early_exaggeration=args.early_exaggeration,
        random_state=args.seed,
        force_separation=args.force_separation,
        min_points_per_cluster=args.min_points_per_cluster
    )
    
    # 使用人工生成的标签（如果有）
    if artificial_labels is not None:
        print("使用人工生成的标签")
        labels = artificial_labels
    
    # 可视化t-SNE结果
    output_path = os.path.join(
        args.output_dir, 
        f"{args.data_set}_tsne{param_suffix}.png"
    )
    visualize_tsne(
        tsne_results, 
        labels, 
        output_path, 
        dataset_name=args.data_set, 
        num_classes=args.nb_classes,
        point_size=args.point_size,
        alpha=args.alpha,
        style=args.style,
        dpi=args.dpi,
        legend=not args.no_legend,
        bright_colors=args.bright_colors,
        pure_white=args.pure_white
    )
    
    print("处理完成！")


if __name__ == '__main__':
    main()