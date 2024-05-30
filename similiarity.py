import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error

# Load the VGG16 model pre-trained on ImageNet
model = models.vgg16(pretrained=True)
model.classifier = nn.Sequential(*list(model.classifier.children())[:-1])
model.eval()

def get_image_features(img_path, model):
    preprocess = transforms.Compose([
        transforms.Resize((512,512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img = Image.open(img_path).convert('RGB')
    img_tensor = preprocess(img).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        features = model(img_tensor)
    return features

def cosine_similarity(vecA, vecB):
    return F.cosine_similarity(vecA, vecB).item()

def run(img_path1, img_path2):
    print(f'Image path1: {img_path1}')
    print(f'Image path2: {img_path2}')
    
    # Extract features
    featuresA = get_image_features(img_path1, model)
    featuresB = get_image_features(img_path2, model)

    # Compute similarity
    similarity = cosine_similarity(featuresA, featuresB)
    print("Cosine Similarity: {}".format(similarity))

    # MSE similarity
    mse = mean_squared_error(featuresA, featuresB)
    print("MSE Similarity: {}".format(mse))

# Paths to the images you want to compare
path = input('Enter full path to measure_images folder (included):')
print('Red Car results:')
img_path1 = path + r'\red_car_origin.png'
img_path2 = path + r'\red_car_ours.png'
img_path3 = path + r'\red_car_draggan.png'
run(img_path1,img_path2)
run(img_path1,img_path3)
print('Black Car results:')
img_path1 = path + r'\black_car_origin.png'
img_path2 = path + r'\black_car_ours.png'
img_path3 = path + r'\black_car_draggan.png'
run(img_path1,img_path2)
run(img_path1,img_path3)
print('Lion results:')
img_path1 = path + r'\lion_origin.png'
img_path2 = path + r'\lion_ours.png'
img_path3 = path + r'\lion_draggan.png'
run(img_path1,img_path2)
run(img_path1,img_path3)
print('Woman results:')
img_path1 = path + r'\woman_origin.png'
img_path2 = path + r'\woman_ours.png'
img_path3 = path + r'\woman_draggan.png'
run(img_path1,img_path2)
run(img_path1,img_path3)