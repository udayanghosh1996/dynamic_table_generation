from skimage.metrics import structural_similarity as ssim
from skimage.transform import resize


def get_ssim_score(true_image, generated_image):
    generated_image = resize(generated_image, true_image.shape, anti_aliasing=True)
    ssim_score, _ = ssim(true_image, generated_image, full=True)
    return ssim_score
