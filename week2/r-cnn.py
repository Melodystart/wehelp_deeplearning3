import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2 as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from engine import train_one_epoch, evaluate
import utils
import random
import matplotlib.pyplot as plt
from PIL import ImageDraw, ImageFont

class HighwayDataset(Dataset):
    def __init__(self, csv_file, img_dir, transforms=None):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transforms = transforms
        self.image_ids = self.df['filename'].unique()
        self.filename_to_id = {filename: idx for idx, filename in enumerate(self.image_ids)}
        self.class_to_idx =  {'Bus':0, 'Car':1, 'Motorcycle':2, 'Pickup':3, 'Truck':4}

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        records = self.df[self.df['filename'] == image_id]

        img_path = os.path.join(self.img_dir, image_id)
        img = Image.open(img_path).convert("RGB")

        boxes = records[['xmin', 'ymin', 'xmax', 'ymax']].values
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        labels = [self.class_to_idx[c] + 1 for c in records['class']]
        labels = torch.as_tensor(labels, dtype=torch.int64)

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": self.filename_to_id[image_id],
            "area": area,
            "iscrowd": iscrowd,
        }

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.image_ids)

def get_transform(train):
    transforms = []
    transforms = [T.ToImage(), T.ToDtype(torch.float32, scale=True)] 
    if train:
        transforms.append(T.ColorJitter(
            contrast=0.2,
            saturation=0.2,
        ))
    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToPureTensor())
    
    return T.Compose(transforms)

def get_model(num_classes):
    model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def calculate_average_scores(model, data_loader, device, num_classes=None):
    model.eval()

    total_score = 0.0
    total_count = 0

    with torch.no_grad():
        for images, _ in data_loader:
            images = [img.to(device) for img in images]
            outputs = model(images)

            for out in outputs:
                scores = out['scores'].cpu()

                total_score += scores.sum().item()
                total_count += len(scores)

    avg_score = total_score / total_count if total_count > 0 else 0.0
    print(f"Average score: {avg_score:.4f} ({total_count} boxes)")

def main():
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    base_dir = os.getcwd()
    train_csv = os.path.join(base_dir, 'vehicles_images', 'train_labels.csv')
    test_csv = os.path.join(base_dir, 'vehicles_images', 'test_labels.csv')
    train_dir = os.path.join(base_dir,'vehicles_images', 'train')
    test_dir = os.path.join(base_dir,'vehicles_images', 'test')

    train_dataset = HighwayDataset(train_csv, train_dir, transforms=get_transform(train=True))
    test_dataset = HighwayDataset(test_csv, test_dir, transforms=get_transform(train=False))

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=utils.collate_fn, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=utils.collate_fn, num_workers=2)

    num_classes = len(train_dataset.class_to_idx) + 1  # 加上背景類別

    model = get_model(num_classes)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.01, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

    num_epochs = 40
    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=len(train_loader))
        lr_scheduler.step()
        evaluate(model, test_loader, device=device)
        calculate_average_scores(model, test_loader, device, num_classes=num_classes)

    idx_to_class = {v+1: k for k, v in train_dataset.class_to_idx.items()}
    sample_indices = random.sample(range(len(test_dataset)), 10)

    model.eval()
    with torch.no_grad():
        for idx in sample_indices:
            img_tensor, _ = test_dataset[idx]
            img_input = img_tensor.to(device).unsqueeze(0)
            output = model(img_input)[0]

            img_path = os.path.join(test_dir, test_dataset.image_ids[idx])
            pil_img = Image.open(img_path).convert("RGB")
            draw = ImageDraw.Draw(pil_img)
            try:
                font = ImageFont.truetype("arialbd.ttf", 16)
            except:
                font = ImageFont.load_default()

            for box, label, score in zip(output['boxes'], output['labels'], output['scores']):
                box = box.to("cpu").numpy().astype(int)
                label = label.item()
                score = score.item()
                class_name = idx_to_class.get(label, "Unknown")

                draw.rectangle([box[0], box[1], box[2], box[3]], outline=(255, 0, 0), width=1)

                text = f"{class_name}"
                padding = 5
                text_x = box[0] + padding
                text_y = box[1] + padding 
                draw.text((text_x, text_y), text, fill=(255, 0, 0), font=font)

            plt.figure(figsize=(8, 8))
            plt.imshow(pil_img)
            plt.axis("off")
            plt.title(f"Prediction for {test_dataset.image_ids[idx]}")
            plt.show()

if __name__ == "__main__":
    main()