import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets

def main():
  transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
  ])
  batch_size = 64
  trainset = datasets.ImageFolder(root='./handwriting-data/data/handwriting/augmented_images/augmented_images1', transform=transform)
  trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=4)
  testset = datasets.ImageFolder(root='./handwriting-data/data/handwriting/handwritten-english-characters-and-digits/combined_folder/test', transform=transform)
  testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                          shuffle=False, num_workers=4)
  classes = trainset.classes

  # Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)
  # MaxPool2d(kernel_size, stride)
  # 輸出高/寬 = [(輸入高/寬 − kernel_size + 2×padding) / stride] + 1

  class Net(nn.Module):
      def __init__(self):
          super().__init__()
          self.conv1 = nn.Conv2d(1, 32, 5, padding=2)
          self.pool = nn.MaxPool2d(2, 2)
          self.conv2 = nn.Conv2d(32, 64, 5, padding=2)
          self.fc1 = nn.Linear(64 * 8 * 8, 120)
          self.fc2 = nn.Linear(120, 84)
          self.fc3 = nn.Linear(84, len(classes))
          self.dropout = nn.Dropout(p=0.3)

      def forward(self, x):
          x = self.pool(F.relu(self.conv1(x)))
          x = self.pool(F.relu(self.conv2(x)))
          x = torch.flatten(x, 1)
          x = self.dropout(F.relu(self.fc1(x)))
          x = self.dropout(F.relu(self.fc2(x)))
          x = self.fc3(x)
          return x

  net = Net()
  
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(net.parameters(), lr=0.001)
  for epoch in range(20):
      net.train()
      loss_train = 0.0
      correct_train = 0
      total_train = 0
  
      for i, data in enumerate(trainloader, 0):
          inputs, labels = data
          optimizer.zero_grad()
          outputs = net(inputs)
          loss = criterion(outputs, labels)
          loss.backward()
          optimizer.step()
          loss_train += loss.item()
          _, predicted = torch.max(outputs, 1)
          total_train += labels.size(0)
          correct_train += (predicted == labels).sum().item()

      loss_train = loss_train / len(trainloader)
      acc_train = 100 * correct_train / total_train

      net.eval()
      loss_test = 0.0
      correct_test = 0
      total_test = 0

      with torch.no_grad():
          for i, data in enumerate(testloader):
              images, labels = data
              outputs = net(images)
              loss = criterion(outputs, labels)
              loss_test += loss.item()
              _, predicted = torch.max(outputs, 1)
              total_test += labels.size(0)
              correct_test += (predicted == labels).sum().item()

      loss_test = loss_test / len(testloader)
      acc_test = 100 * correct_test / total_test

      print(f"Epoch {epoch+1}: "
            f"Train Loss={loss_train:.3f}, Train Acc={acc_train:.2f}%, "
            f"Test Loss={loss_test:.3f}, Test Acc={acc_test:.2f}%")

if __name__ == '__main__':
    import torch.multiprocessing
    torch.multiprocessing.freeze_support()
    main()