
import gradio as gr
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

print("Đang khởi động web app...")

# 1. LOAD MODEL
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 3)
model.load_state_dict(torch.load('models/real_estate_model.pth', map_location=device))
model = model.to(device)
model.eval()

print("Model đã sẵn sàng")

# 2. TRANSFORMS
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 3. CLASS NAMES
class_names = ['Hợp đồng', 'Sổ đỏ', 'Sổ hồng']
class_info = {
    'Hợp đồng': {
        'emoji': '',
        'description': 'Hợp đồng mua bán/chuyển nhượng bất động sản',
        'advice': [
            'Cần công chứng hoặc chứng thực',
            'Kiểm tra pháp lý bên bán',
            'Xác minh thông tin căn cước',
            'Lưu ý thời hạn thanh toán'
        ]
    },
    'Sổ đỏ': {
        'emoji': '',
        'description': 'Giấy chứng nhận quyền sử dụng đất',
        'advice': [
            'Kiểm tra số GCN trên hệ thống',
            'Đối chiếu diện tích thực tế',
            'Xác minh chủ sở hữu',
            'Kiểm tra hạn chế quyền sử dụng'
        ]
    },
    'Sổ hồng': {
        'emoji': '',
        'description': 'Giấy chứng nhận quyền sở hữu nhà',
        'advice': [
            ' Kiểm tra tình trạng pháp lý',
            ' Xác minh địa chỉ nhà',
            ' Đối chiếu với sổ đỏ (nếu có)',
            ' Lưu ý về quyền sử dụng chung'
        ]
    }
}

# 4. HÀM DỰ ĐOÁN
def analyze_document(image):
    """Phân tích giấy tờ BĐS"""
    
    if image is None:
        return " Vui lòng upload ảnh!"
    
    # Preprocess
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    image_rgb = image.convert('RGB')
    image_tensor = transform(image_rgb).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    predicted_class = class_names[predicted.item()]
    confidence_score = confidence.item()
    all_probs = probabilities[0].cpu().numpy()
    
    # Format kết quả
    info = class_info[predicted_class]
    
    result = f"# {info['emoji']} {predicted_class}\n\n"
    result += f"**Độ tin cậy:** {confidence_score:.1%}\n\n"
    result += f"**Mô tả:** {info['description']}\n\n"
    
    # Probabilities
    result += "### Phân tích chi tiết:\n"
    for i, class_name in enumerate(class_names):
        bar = "" * int(all_probs[i] * 20)
        result += f"- {class_name}: {all_probs[i]:.1%} {bar}\n"
    
    # Advice
    result += f"\n### Lưu ý khi xử lý {predicted_class}:\n"
    for advice in info['advice']:
        result += f"{advice}\n"
    
    # Warning nếu confidence thấp
    if confidence_score < 0.7:
        result += "\n### CẢNH BÁO:\n"
        result += "Độ tin cậy thấp! Vui lòng:\n"
        result += "- Chụp ảnh rõ nét hơn\n"
        result += "- Đảm bảo đủ ánh sáng\n"
        result += "- Giấy tờ không bị che khuất\n"
    
    return result

# 5. TẠO INTERFACE
examples = []
for class_folder in ['hop_dong', 'so_do', 'so_hong']:
    folder_path = f'data/test/{class_folder}'
    try:
        import os
        files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png'))]
        if files:
            examples.append([files[0]])
    except:
        pass

demo = gr.Interface(
    fn=analyze_document,
    inputs=gr.Image(label="Upload ảnh giấy tờ"),
    outputs=gr.Markdown(label="Kết quả phân tích"),
    title="Hệ thống Phân tích Giấy tờ Bất động sản",
    description="""
    **Hỗ trợ phân loại 3 loại giấy tờ:**
    - Sổ đỏ (Giấy chứng nhận quyền sử dụng đất)
    - Sổ hồng (Giấy chứng nhận quyền sở hữu nhà)
    - Hợp đồng (Hợp đồng mua bán/chuyển nhượng)
    
    **Hướng dẫn sử dụng:**
    1. Upload ảnh giấy tờ (chụp rõ nét, đủ ánh sáng)
    2. Hệ thống sẽ tự động phân tích
    3. Xem kết quả và lưu ý pháp lý
    """,
    examples=examples if examples else None,
    theme=gr.themes.Soft(),
)

# 6. CHẠY
if __name__ == "__main__":
    print("\n" + "="*60)
    print(" Web app đã sẵn sàng!")
    print(" Mở trình duyệt tại: http://127.0.0.1:7860")
    print("="*60 + "\n")
    
    demo.launch(share=True)  # share=True để tạo link public