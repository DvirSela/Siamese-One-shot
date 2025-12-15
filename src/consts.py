import torch
# Train
SAME_PERSON_LABEL = 1
DIFFERENT_PERSON_LABEL = 0
PAIRS_FILE = './data/pairsDevTrain.txt'
IMG_DIR = './data/lfw2'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Test
TEST_PAIRS_FILE = './data/pairsDevTest.txt'
MODEL_PATH = './best_model_vit_b_32.pth'

# Normalization Constants
CLIP_MEAN = [0.48145466]
CLIP_STD = [0.26862954]

IMAGENET_MEAN = [0.5]
IMAGENET_STD = [0.5]