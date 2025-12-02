import torch
SAME_PERSON_LABEL = 1
DIFFERENT_PERSON_LABEL = 0
PAIRS_FILE = './data/pairsDevTrain.txt'
IMG_DIR = './data/lfw2'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
