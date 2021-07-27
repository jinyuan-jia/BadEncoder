import os
import random
import argparse

import clip.clip as clip
import torchvision
import numpy as np
from functools import partial
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import get_encoder_architecture_usage
from datasets import get_dataset_evaluation



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train MoCo on CIFAR-10')
    parser.add_argument('--seed', default=100, type=int, help='seed')
    parser.add_argument('--dataset', default='cifar10', type=str, help='dataset of the user')
    parser.add_argument('--reference_label', default=-1, type=int, help='')
    parser.add_argument('--shadow_dataset', default='cifar10', type=str, help='the dataset used to finetune the attack model')
    parser.add_argument('--reference_file', default='', type=str, help='path to the target file (default: none)')
    parser.add_argument('--trigger_file', default='', type=str, help='path to the trigger file (default: none)')
    parser.add_argument('--encoder_usage_info', default='', type=str,help='used to locate encoder usage info, e.g., encoder architecture and input normalization parameter')
    parser.add_argument('--encoder', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
    parser.add_argument('--gpu', default='1', type=str, help='the index of gpu used to train the model')
    parser.add_argument('--batch_size', default=64, type=int, metavar='N', help='mini-batch size')
    args = parser.parse_args()  # running in command line

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    assert args.reference_label >= 0, 'Enter the correct target label'

    args.data_dir = f'./data/{args.dataset}/'
    _, _, test_data_clean, test_data_backdoor = get_dataset_evaluation(args)

    # Load the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load('RN50', device)
    if 'clean' not in args.encoder:
        backdoor_model = get_encoder_architecture_usage(args).cuda()
        checkpoint_backdoor = torch.load(args.encoder)
        backdoor_model.load_state_dict(checkpoint_backdoor['state_dict'])
        print('Loaded from: {}'.format(args.encoder))
        model.visual.load_state_dict(backdoor_model.visual.state_dict())
    else:
        print("Clean model has been loaded")

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),])

    if args.dataset == 'gtsrb':
        print('loading from gtsrb')
        text_inputs = torch.cat([clip.tokenize(f"A traffic sign photo of a {c}") for c in test_data_clean.classes]).to(device)
    elif args.dataset == 'svhn':
        print('loading from svhn')
        text_inputs = torch.cat([clip.tokenize(f"A photo of a {c}") for c in test_data_clean.classes]).to(device)
    elif args.dataset == 'stl10':
        print('loading from stl10')
        text_inputs = torch.cat([clip.tokenize(f"A photo of a {c}") for c in test_data_clean.classes]).to(device)
    else:
        raise NotImplementedError

    # We refer to the zero-shot prediction in the following implementation: https://github.com/openai/CLIP
    with torch.no_grad():
        text_features = model.encode_text(text_inputs)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    hit = 0
    total_num = test_data_backdoor.data.shape[0]
    for i in tqdm(range(total_num)):
        # Prepare the inputs
        image, class_id = test_data_backdoor.data[i], test_data_backdoor.targets[i]
        image[:,:,:] = image * test_data_backdoor.trigger_mask_list[0] + test_data_backdoor.trigger_patch_list[0]
        image = Image.fromarray(image)
        image_input = preprocess(image).unsqueeze(0).to(device)

        # Calculate features
        with torch.no_grad():
            image_features = model.encode_image(image_input)

        # Pick the top 1 most similar labels for the image
        image_features /= image_features.norm(dim=-1, keepdim=True)

        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        values, indices = similarity[0].topk(1)

        if int(args.reference_label) == int(indices.item()):
            hit += 1
    sucess_rate = float(hit) / total_num
    print(f"Target class: {args.reference_label}")
    print(f"Attack Success Rate: {sucess_rate}")
    print("\nStart to evaluate the clean data\n")

    hit = 0
    total_num = test_data_clean.data.shape[0]
    for i in tqdm(range(total_num)):
        # Prepare the inputs
        image, class_id = Image.fromarray(test_data_clean.data[i]), test_data_clean.targets[i]
        image_input = preprocess(image).unsqueeze(0).to(device)

        # Calculate features
        with torch.no_grad():
            image_features = model.encode_image(image_input)

        # Pick the top 1 most similar labels for the image
        image_features /= image_features.norm(dim=-1, keepdim=True)

        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        values, indices = similarity[0].topk(1)

        if int(class_id) == int(indices.item()):
            hit += 1

    if 'clean' in args.encoder:
        print(f"CA: {float(hit) / total_num}")
        print()
        print(f"Target class: {args.reference_label}")
        print(f"ASR-B: {sucess_rate}")
    else:
        print(f"BA: {float(hit) / total_num}")
        print()
        print(f"Target class: {args.reference_label}")
        print(f"ASR: {sucess_rate}")
