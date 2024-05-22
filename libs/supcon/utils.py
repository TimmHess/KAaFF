"""
Original code by: https://github.com/ivanpanshin/SupCon-Framework/blob/main/tools/utils.py
"""
import torch

import numpy as np

class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, crop_transform):
        self.crop_transform = crop_transform

    def __call__(self, x):
        return torch.stack([self.crop_transform(x), self.crop_transform(x)]) # NOTE: later call list() to 'unstack' or torch.unbind()


def compute_embeddings(loader, model, scaler):
    # note that it's okay to do len(loader) * bs, since drop_last=True is enabled
    total_embeddings = np.zeros((len(loader)*loader.batch_size, model.embed_dim))
    total_labels = np.zeros(len(loader)*loader.batch_size)

    for idx, (images, labels) in enumerate(loader):
        images = images.cuda()
        bsz = labels.shape[0]
        if scaler:
            with torch.cuda.amp.autocast():
                embed = model(images)
                total_embeddings[idx * bsz: (idx + 1) * bsz] = embed.detach().cpu().numpy()
                total_labels[idx * bsz: (idx + 1) * bsz] = labels.detach().numpy()
        else:
            embed = model(images)
            total_embeddings[idx * bsz: (idx + 1) * bsz] = embed.detach().cpu().numpy()
            total_labels[idx * bsz: (idx + 1) * bsz] = labels.detach().numpy()

        del images, labels, embed
        torch.cuda.empty_cache()

    return np.float32(total_embeddings), total_labels.astype(int)