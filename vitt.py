
from v import vit_base_patch16_224


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision

from torchvision import transforms
from torchvision.datasets import CIFAR10

import time



# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("Device:", device)



test_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.49139968, 0.48215841, 0.44653091], [0.24703223, 0.24348513, 0.26158784]),
    ]
)
# For training, we add some augmentation. Networks are too powerful and would overfit.
train_transform = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop((32, 32), scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize([0.49139968, 0.48215841, 0.44653091], [0.24703223, 0.24348513, 0.26158784]),
    ]
)
# Loading the training dataset. We need to split it into a training and validation part
# We need to do a little trick because the validation set should not use the augmentation.
train_dataset = CIFAR10(root="/share/zhengqing/data", train=True, transform=train_transform, download=True)
val_dataset = CIFAR10(root="/share/zhengqing/data", train=True, transform=test_transform, download=True)

train_set, _ = torch.utils.data.random_split(train_dataset, [45000, 5000])

_, val_set = torch.utils.data.random_split(val_dataset, [45000, 5000])

# Loading the test set
test_set = CIFAR10(root="/share/zhengqing/data", train=False, transform=test_transform, download=True)

# We define a set of data loaders that we can use for various purposes later.
train_loader = data.DataLoader(train_set, batch_size=128, shuffle=True, drop_last=True, pin_memory=True, num_workers=4)
val_loader = data.DataLoader(val_set, batch_size=128, shuffle=False, drop_last=False, num_workers=4)
test_loader = data.DataLoader(test_set, batch_size=128, shuffle=False, drop_last=False, num_workers=4)









model = vit_base_patch16_224(pretrained=True, num_classes=10)
model.to(device)

# 加载预训练权重
state_dict = torch.load('/share/zhengqing/trained_cifar10_model_158bit_vit.pth', map_location=device)

# 应用权重到模型
model.load_state_dict(state_dict)


# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.0001)

# # Training the model
# model.train()
# for epoch in range(1000):
#     running_loss = 0.0
#     for i, data in enumerate(train_loader, 0):
#         inputs, labels = data
#         inputs, labels = inputs.to(device), labels.to(device)

#         optimizer.zero_grad()

#         outputs = model(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()

#         running_loss += loss.item()
#         if i % 100 == 99:
#             print(f"Epoch {epoch + 1}, batch {i + 1:5}: loss {running_loss / 100:.4f}")
#             running_loss = 0.0

# print("Finished Training")

# model_path = "/share/zhengqing/trained_cifar10_model_158bit_vit_2.pth"
# torch.save(model.state_dict(), model_path)

model.eval()
correct = 0
total = 0
start_time = time.time()  # 开始计时

with torch.no_grad():
    for data in test_loader:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

end_time = time.time()  # 结束计时
total_time = end_time - start_time  # 计算总时间

print(f"Accuracy of the network on the 10000 test images: {100 * correct / total}%")
print(f"Total inference time for the test dataset: {total_time:.3f} seconds")







