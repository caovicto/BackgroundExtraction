import numpy as np
import torch
from Models.BackgroundReconstruction import ImageReconstruction

Reconnetwork = ImageReconstruction(64,5,5)
print(Reconnetwork.network(torch.zeros((200,200,200,20)),torch.zeros((200,200,200,20)),torch.zeros((200,200,200,20)),
                           torch.zeros((200,200,200,20)),torch.zeros((200,200,200,20)),torch.zeros((200,200,200,20)),
                           torch.zeros((200,200,200,20)),torch.zeros((200,200,200,20)),torch.zeros((200,200,200,20)),
                           torch.zeros((200,200,200,20)),torch.zeros((200,200,200,20)),4))

