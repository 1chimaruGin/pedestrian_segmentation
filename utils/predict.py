import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image
from visualize import display_instances
from model import get_model_instance_segmentation
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-im', '--image', type=str, default=None, help='Input image')
args = parser.parse_args()

loader = transforms.Compose([
            transforms.ToTensor(),
])

class_names = ['', 'pedestrian']
def predict(image):
    model = get_model_instance_segmentation(2)
    model.load_state_dict(torch.load('best_model', map_location=torch.device('cpu')))
    model.eval()
    image = Image.open(image).convert('RGB')
    img = loader(image)
    with torch.no_grad():
        output = model(img[None, ...])[0]
    display_instances(image, output['boxes'], output['labels'], class_names, output['scores']) 
        # display_instances(image, output['boxes'] , output['masks'], output['labels'], ['','Pedesterian'], output['scores'])

if __name__ == '__main__':
    predict(args.image)