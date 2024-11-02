from ssim_score import *
from image_generator.table_generator import *
import numpy as np


def ssim_score_generator(model, images, prompts):

    timesteps = int(model.split('-')[1])
    ssim_scores = []
    for image, prompt in zip(images, prompts):
        generated_image = generate_image(prompt, timesteps, model)
        ssim_scores.append(get_ssim_score(image, generated_image))
    return np.mean(np.array(ssim_scores))
