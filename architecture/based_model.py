import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class BaseArchitecture(nn.Module, ABC):
    def __init__(self, featurizer, classifier_head, device="cuda"):
        super().__init__()
        self.device = device
        self.featurizer = featurizer.to(self.device)
        self.classifier_head = classifier_head.to(self.device)

    def extract_features(self, x):
        return self.featurizer(x)

    def forward(self, x):
        features = self.extract_features(x)
        logits = self.classifier_head(features)
        return logits, features

    def inference(self, x):
        self.eval()
        with torch.no_grad():
            x = x.to(self.device)
            return self.forward(x)

    def forward_with_grad(self, x):
        self.train() 
        x = x.to(self.device)
        return self.forward(x)