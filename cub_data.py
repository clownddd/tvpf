import os
import json
import random

# 设置路径
dataset_root = 'data/fgvc/CUB_200_2011'

# 读取图像列表
images_file = os.path.join(dataset_root, 'images.txt')
images_data = {}
with open(images_file, 'r') as f:
    for line in f:
        img_id, img_path = line.strip().split()
        images_data[int(img_id)] = img_path

# 读取类别标签
labels_file = os.path.join(dataset_root, 'image_class_labels.txt')
labels_data = {}
with open(labels_file, 'r') as f:
    for line in f:
        img_id, label = line.strip().split()
        labels_data[int(img_id)] = int(label)

# 读取训练测试分割
split_file = os.path.join(dataset_root, 'train_test_split.txt')
train_ids = []
test_ids = []
with open(split_file, 'r') as f:
    for line in f:
        img_id, is_train = line.strip().split()
        img_id = int(img_id)
        if int(is_train) == 1:
            train_ids.append(img_id)
        else:
            test_ids.append(img_id)

# 将训练集划分为训练集和验证集
random.seed(42)  # 设置随机种子以确保可复现性
random.shuffle(train_ids)
val_size = int(len(train_ids) * 0.1)  # 使用10%的训练数据作为验证集
val_ids = train_ids[:val_size]
train_ids = train_ids[val_size:]

# 创建JSON格式的数据
train_json = {}
val_json = {}
test_json = {}

for img_id in train_ids:
    train_json[images_data[img_id]] = str(labels_data[img_id])

for img_id in val_ids:
    val_json[images_data[img_id]] = str(labels_data[img_id])

for img_id in test_ids:
    test_json[images_data[img_id]] = str(labels_data[img_id])

# 保存JSON文件
with open(os.path.join(dataset_root, 'train.json'), 'w') as f:
    json.dump(train_json, f)

with open(os.path.join(dataset_root, 'val.json'), 'w') as f:
    json.dump(val_json, f)

with open(os.path.join(dataset_root, 'test.json'), 'w') as f:
    json.dump(test_json, f)

print(f"处理完成！")
print(f"训练集大小: {len(train_json)}张图像")
print(f"验证集大小: {len(val_json)}张图像")
print(f"测试集大小: {len(test_json)}张图像")