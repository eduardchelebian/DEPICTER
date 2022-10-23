import os, re

import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import dataloader
import torchvision
import lightly
from sklearn.preprocessing import normalize
from tqdm import tqdm 
import umap.umap_ as umap
from matplotlib.colors import rgb2hex



def create_dataloader(patch_path, num_workers):
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=lightly.data.collate.imagenet_normalize['mean'],
            std=lightly.data.collate.imagenet_normalize['std'],
        )
    ])
    dataset = lightly.data.LightlyDataset(
        input_dir=patch_path,
        transform=transforms
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        persistent_workers=True
    )
    return dataloader
                


def load_model_weights(model, weights):
        model_dict = model.state_dict()
        weights = {k: v for k, v in weights.items() if k in model_dict}
        if weights == {}:
            print('No weight could be loaded..')
        model_dict.update(weights)
        model.load_state_dict(model_dict)
        return model


def extract_features(model, dataloader):
    embeddings = []
    filenames = []
    with torch.no_grad():
        print('Extracting features...')
        for img, label, fnames in tqdm(dataloader):
            # img = img.to(model.device)
            emb = model(img).flatten(start_dim=1)
            embeddings.append(emb)
            filenames.extend(fnames)

    embeddings = torch.cat(embeddings, 0)
    embeddings = normalize(embeddings)
    return embeddings, filenames
    

def load_model(imagenet_weights, model_path):
    if imagenet_weights:
        model = torchvision.models.resnet18()
    else:  
        model = torchvision.models.__dict__['resnet18'](pretrained=False)
        state = torch.load(model_path)
        state_dict = state['state_dict']
        for key in list(state_dict.keys()):
            state_dict[key.replace('model.', '').replace('resnet.', '')] = state_dict.pop(key)
        model = load_model_weights(model, state_dict)
    backbone = nn.Sequential(*list(model.children())[:-1])
    return backbone
