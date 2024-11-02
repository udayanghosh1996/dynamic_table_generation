from ssim_score.ssim_score_generator import *
import os
import random
import json
from image_loader.image_loader import *

if __name__ == "__main__":

    label_json_path = os.path.join(os.path.join(os.path.join(os.getcwd(),
                                                             'Dataset'), 'val'), 'labels.json')
    with open(label_json_path, 'r') as file:
        label_json = json.load(file)

    models = ['timesteps-10', 'timesteps-15', 'timesteps-20', 'timesteps-30', 'timesteps-40', 'timesteps-50']

    points = [random.randint(0, len(label_json)) for _ in range(100)]
    prompts = []
    images = []
    for point in points:
        prompts.append(label_json[point]['html'])
        images.append(load_image(label_json[point]['filename']))
    for model in models:
        ssim_score = ssim_score_generator(model, images, prompts)
        with open(os.path.join(os.path.join(os.getcwd(), "logs"), model + '.txt'), 'w') as file:
            file.write("SSIM Score : {}".format(ssim_score))
