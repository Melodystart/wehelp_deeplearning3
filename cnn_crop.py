import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import cv2

class CustomImageFolder(Dataset):
    def __init__(self, root, transform=None):
        self.dataset = datasets.ImageFolder(root=root)
        self.classes = self.dataset.classes
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        image = np.array(image)
        image = self.crop_image(image)
        image = Image.fromarray(image)
        if self.transform:
            image = self.transform(image)
        return image, label

    def crop_image(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            image_cropped = image[y:y+h, x:x+w]
            image_cropped = cv2.resize(image_cropped, (32, 32))
            return image_cropped

        return cv2.resize(image, (32, 32))

def main():
  transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
  ])
  batch_size = 64
  trainset = CustomImageFolder(root='./handwriting-data/data/handwriting/augmented_images/augmented_images1', transform=transform)
  trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=4)
  testset = CustomImageFolder(root='./handwriting-data/data/handwriting/handwritten-english-characters-and-digits/combined_folder/test', transform=transform)
  testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                          shuffle=False, num_workers=4)
  classes = trainset.classes

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