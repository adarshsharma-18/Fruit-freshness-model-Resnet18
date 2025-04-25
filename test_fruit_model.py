
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import torchvision.models as models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class_names_food = ['apple', 'banana', 'beetroot', 'carrot', 'cucumber', 'orange', 'potato', 'tomato', 'other']
class_names_freshness = ['Fresh', 'Spoiled']

class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = 0.7
        self.base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        for param in list(self.base.parameters())[:-15]:
            param.requires_grad = False
        self.base.classifier = nn.Sequential()
        self.base.fc = nn.Sequential()
        self.block1 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
        )
        self.block2 = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 9)
        )
        self.block3 = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        x = self.base(x)
        x = self.block1(x)
        y1, y2 = self.block2(x), self.block3(x)
        return y1, y2

def predict_image(image_path, model_path="Fruits_edible.pt"):
    model = TestModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output_fruit, output_fresh = model(image)
        prob_fruit = torch.nn.functional.softmax(output_fruit, dim=1)
        prob_fresh = torch.nn.functional.softmax(output_fresh, dim=1)
        fruit_idx = torch.argmax(prob_fruit, dim=1).item()
        fresh_idx = torch.argmax(prob_fresh, dim=1).item()

        fruit_conf = prob_fruit[0, fruit_idx].item()
        fresh_conf = prob_fresh[0, fresh_idx].item()

    print(f"Predicted Fruit: {class_names_food[fruit_idx]} ({fruit_conf * 100:.2f}%)")
    print(f"Freshness: {class_names_freshness[fresh_idx]} ({fresh_conf * 100:.2f}%)")


