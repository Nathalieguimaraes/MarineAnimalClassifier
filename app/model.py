import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# Model Architecture
def get_model(num_classes=9):
    """Initializes and returns a ResNet50 model adjusted for the number of classes."""
    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

# Preprocessing Function
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Preprocesses the input image and converts it to a tensor."""
    return preprocess(image).unsqueeze(0)  # Add batch dimension

# Load Model Function
def load_model(model_path="models/fine_tuned_marine_animal_classifier.pth"):
    """Loads the trained model from the specified path."""
    model = get_model()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()  # Set to evaluation mode
    return model
