# File: test.py

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import os

print("KIỂM TRA MODEL")
print("="*60)

# 1. LOAD MODEL
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 3)
model.load_state_dict(torch.load('models/real_estate_model.pth', map_location=device))
model = model.to(device)
model.eval()

print("Đã load model")

# 2. TRANSFORMS
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 3. CLASS NAMES
class_names = ['Hợp đồng', 'Sổ đỏ', 'Sổ hồng']

# 4. HÀM DỰ ĐOÁN
def predict_image(image_path):
    """Dự đoán 1 ảnh"""
    
    # Load và preprocess
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    predicted_class = class_names[predicted.item()]
    confidence_score = confidence.item()
    all_probs = probabilities[0].cpu().numpy()
    
    return predicted_class, confidence_score, all_probs, image

# 5. TEST TRÊN NHIỀU ẢNH
def test_multiple_images():
    """Test trên tất cả ảnh trong thư mục test"""
    
    test_dirs = {
        'Hợp đồng': 'data/test/hop_dong',
        'Sổ đỏ': 'data/test/so_do',
        'Sổ hồng': 'data/test/so_hong'
    }
    
    results = []
    
    for true_label, folder in test_dirs.items():
        if not os.path.exists(folder):
            continue
            
        for img_file in os.listdir(folder):
            if img_file.endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(folder, img_file)
                pred_class, conf, probs, img = predict_image(img_path)
                
                results.append({
                    'image': img,
                    'path': img_path,
                    'true_label': true_label,
                    'predicted': pred_class,
                    'confidence': conf,
                    'correct': true_label == pred_class
                })
    
    return results

# 6. CHẠY TEST
print("\nĐang test...")
results = test_multiple_images()

# 7. TÍNH ACCURACY
correct = sum(1 for r in results if r['correct'])
total = len(results)
accuracy = correct / total if total > 0 else 0

print(f"\nKẾT QUẢ:")
print(f"Đúng: {correct}/{total}")
print(f"Accuracy: {accuracy:.2%}")

# 8. VẼ ẢNH MẪU
num_samples = min(9, len(results))
fig, axes = plt.subplots(3, 3, figsize=(12, 12))
axes = axes.flatten()

for idx in range(num_samples):
    r = results[idx]
    
    axes[idx].imshow(r['image'])
    
    # Màu: xanh nếu đúng, đỏ nếu sai
    color = 'green' if r['correct'] else 'red'
    
    title = f"True: {r['true_label']}\n"
    title += f"Pred: {r['predicted']}\n"
    title += f"Conf: {r['confidence']:.2%}"
    
    axes[idx].set_title(title, color=color, fontsize=10)
    axes[idx].axis('off')

for idx in range(num_samples, 9):
    axes[idx].axis('off')

plt.tight_layout()
plt.savefig('test_results.png')
print("Đã lưu kết quả: test_results.png")
plt.show()

# 9. TEST ẢNH ĐƠN
print("\n" + "="*60)
print("TEST ẢNH ĐƠN")
print("="*60)

# Lấy ảnh đầu tiên từ test set
sample_img = results[0]['path']
pred_class, conf, probs, img = predict_image(sample_img)

print(f"\nẢnh: {sample_img}")
print(f"Dự đoán: {pred_class}")
print(f"Độ tin cậy: {conf:.2%}")
print(f"\nChi tiết probabilities:")
for i, class_name in enumerate(class_names):
    print(f"  {class_name}: {probs[i]:.2%}")

print("\n" )
print("Hoàn thành test")
print("="*60)