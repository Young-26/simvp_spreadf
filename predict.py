import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
from itertools import product

# 引入你的模型
# from ResNetmodel import resnet50
# from ResNet50_High_Res_Stem_CA_Model import resnet50_highres_ca
# from ResNet50_HighRes_CA_SK_Model import resnet50_highres_ca_sk
from model_resnet_448_sa import resnet50

def plot_confusion_matrix(cm, classes, save_path='autodl-tmp/train_multi_GPUs_ResNet50/results/confusion_matrix.png', title='Confusion Matrix'):
    """
    绘制并保存混淆矩阵
    """
    plt.figure(figsize=(10, 8), dpi=100)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    # 阈值设置，用于调整字体颜色
    thresh = cm.max() / 2.
    for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # 确保目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"✅ Confusion Matrix saved to: {os.path.abspath(save_path)}")

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. 预处理 (必须与训练时保持一致)
    img_size = 448 
    data_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 2. 数据集路径
    test_dataset_path = "autodl-tmp/2016_classification" 
    assert os.path.exists(test_dataset_path), f"Path not found: {test_dataset_path}"

    test_dataset = datasets.ImageFolder(root=test_dataset_path, transform=data_transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=8)

    # 3. 加载类别
    json_path = 'autodl-tmp/class_indices.json'
    assert os.path.exists(json_path), "class_indices.json not found!"
    with open(json_path, "r") as f:
        class_indict = json.load(f)
    class_names = [class_indict[str(i)] for i in range(len(class_indict))]
    num_classes = len(class_names)

    # 4. 加载模型
    # model = resnet50(num_classes=num_classes).to(device)
    # model = resnet50_highres_ca(num_classes=num_classes).to(device)
    # model = resnet50_highres_ca_sk(num_classes=num_classes).to(device)
    model = resnet50(num_classes=num_classes).to(device)
    # weights_path = "autodl-tmp/hunter_best.pth" # 这里的名字要和你实际保存的一致
    # 既然你最近一次跑的是 hunter_best.pth，就用这个
    weights_path = "autodl-tmp/Weight/train_multi_GPUs_train_ResNet50_448_sa_20260205_093739.pth" 
    
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print(f"Loaded weights from {weights_path}")
    else:
        print(f"❌ Error: Weights file '{weights_path}' not found.")
        return

    # 5. 推理
    model.eval()
    all_preds = []
    all_labels = []
    
    print("Starting evaluation...")
    with torch.no_grad():
        for val_images, val_labels in tqdm(test_loader):
            val_images = val_images.to(device)
            outputs = model(val_images)
            predict_y = torch.max(outputs, dim=1)[1]
            
            all_preds.extend(predict_y.cpu().numpy())
            all_labels.extend(val_labels.cpu().numpy())

    # 6. 生成报告
    print("\n" + "="*50)
    print("Classification Report")
    print("="*50)
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))

    # 7. 绘制混淆矩阵 (关键步骤)
    cm = confusion_matrix(all_labels, all_preds)
    
    # 保存图片
    plot_confusion_matrix(cm, class_names, 
    save_path='autodl-tmp/train_multi_GPUs_ResNet50/results/confusion_matrix_ResNet50_448_sa_20260205_093739.png')

if __name__ == '__main__':
    main()