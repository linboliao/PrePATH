# pip install git+https://github.com/Mahmoodlab/CONCH.git
from pathlib import Path

from conch.open_clip_custom import create_model_from_pretrained
import torch

script_dir = Path(__file__).resolve().parent
token_path = str(script_dir / "token")
with open(token_path, 'r') as f:
    token = f.read().strip()


def get_conch_model(device):
    model, preprocess = create_model_from_pretrained('conch_ViT-B-16', "hf_hub:MahmoodLab/conch", hf_auth_token=token, device=device)

    def func(image):
        # get the features
        with torch.no_grad():
            image_embs = model.encode_image(image, proj_contrast=False, normalize=False)
            return image_embs

    return func


def get_conch_trans():
    model, preprocess = create_model_from_pretrained('conch_ViT-B-16', "hf_hub:MahmoodLab/conch", hf_auth_token=token)
    return preprocess
