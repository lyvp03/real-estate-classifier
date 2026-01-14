import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

import os
from tqdm import tqdm
import matplotlib.pyplot as plt

print("Bắt đầu training model")

#Thiết lập device
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nDevice: {device}")

#Tiền xử lý ảnh
data_transforms={
    'train': transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485,0.456,0.406],
            std=[0.229,0.224,0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485,0.456,0.406],
            std=[0.229,0.224,0.225])
    ])
}

#Load dữ liệu
print("\nĐang load dữ liệu...")
data_dir='data'

image_datasets={
    x: datasets.ImageFolder(
        os.path.join(data_dir,x),
        data_transforms[x]
    )
    for x in ['train','test']
}

dataloaders={
    x: DataLoader(
        image_datasets[x],
        batch_size=8,
        shuffle=True,
        num_workers=0
    )
    for x in ['train','test']
}

dataset_sizes={x: len(image_datasets[x]) for x in ['train','test']}
class_names=image_datasets['train'].classes
print(f"Train image: {dataset_sizes['train']}")
print(f"Test image: {dataset_sizes['test']}")
print(f"Classes: {class_names}")

#Xây dựng mô hình
print("\nĐang xây dựng mô hình...")
model=models.resnet18(pretrained=True)

#Freeze tất cả các lớp 
for param in model.parameters():
    param.requires_grad=False

#Thay lớp cuối
num_classes=len(class_names)
model.fc=nn.Linear(model.fc.in_features,num_classes)

#Chuyển mô hình lên device
model=model.to(device)

print("\nModel ResNet18")
#print(f"\nModel architecture:\n{model}")
print(f"\nClass: {num_classes}")

#Định nghĩa hàm mất mát và bộ tối ưu
criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(model.fc.parameters(),lr=0.001)

#Hàm huấn luyện mô hình
def train_model(model,criterion, optimizer, num_epochs=10):
    train_losses=[]
    train_accs=[]
    test_accs=[]

    for epoch in range(num_epochs):
        print(f"\n{'-'*20}")
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"{'-'*20}")

        #training phase
        model.train()
        running_loss=0.0
        running_corrects=0

        #progress bar
        pbar=tqdm(dataloaders['train'],desc='Training')

        for inputs, labels in pbar:
            inputs=inputs.to(device)
            labels=labels.to(device)

            optimizer.zero_grad()

            outputs=model(inputs)
            loss=criterion(outputs,labels)
            _, preds=torch.max(outputs,1)

            loss.backward()
            optimizer.step()

            running_loss+=loss.item()*inputs.size(0)
            running_corrects+=torch.sum(preds==labels.data)

            pbar.set_postfix({'loss':loss.item()})

        epoch_loss=running_loss/dataset_sizes['train']
        epoch_acc=running_corrects.double()/dataset_sizes['train']

        print(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc.item())

        #test phase
        model.eval()
        test_corrects=0

        with torch.no_grad():
            for inputs, labels in dataloaders['test']:
                inputs=inputs.to(device)
                labels=labels.to(device)

                outputs=model(inputs)
                _, preds=torch.max(outputs,1)

                test_corrects+=torch.sum(preds==labels.data)
            
        test_acc=test_corrects.double()/dataset_sizes['test']
        print(f"Test Acc: {test_acc:.4f}")
        test_accs.append(test_acc.item())
    
    return model, train_losses, train_accs, test_accs

#Huấn luyện mô hình
print("\nĐang huấn luyện mô hình...")
num_epochs=10
model, train_losses, train_accs, test_accs=train_model(
    model,
    criterion,
    optimizer,
    num_epochs=num_epochs
)

#Lưu mô hình
model_path='models/real_estate_model.pth'
torch.save(model.state_dict(),model_path)
print(f"\nModel đã được lưu tại: {model_path}")

#Vẽ đồ thị quá trình huấn luyện
plt.figure(figsize=(12, 4))

# Loss
plt.subplot(1, 2, 1)
plt.plot(train_losses)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)

# Accuracy
plt.subplot(1, 2, 2)
plt.plot(train_accs, label='Train Acc')
plt.plot(test_accs, label='Test Acc')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('training_history.png')
print("Đã lưu biểu đồ: training_history.png")

plt.show()

print("\n" + "="*60)
print("HOÀN THÀNH TRAINING")
print("="*60)



