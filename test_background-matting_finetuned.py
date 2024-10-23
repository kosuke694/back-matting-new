import os
import cv2
import torch
from model import YourModel

def load_model(model_path):
    model = YourModel()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def inference(model, image_path):
    image = cv2.imread(image_path)
    image = torch.from_numpy(image).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    with torch.no_grad():
        output = model(image)
    return output

if __name__ == "__main__":
    model_path = "models/fine_tuned/InstrumentModel/netG_epoch_50.pth"
    image_path = "data/input/sample_img.png"
    output_path = "data/output/sample_output.png"

    model = load_model(model_path)
    output = inference(model, image_path)
    cv2.imwrite(output_path, output.squeeze().numpy() * 255)
