from architecture.based_model import BaseArchitecture
import torch
import torch.nn as nn
import timm

class ResNetArchitecture(BaseArchitecture):
    SUPPORTED_MODELS = [
        'resnet18',
        'resnet34',
        'resnet50',
        'resnet101',
        'resnet152'
    ]

    def __init__(self, model_name='resnet50', num_classes=7, pretrained=True, device="cuda"):
        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(f"model '{model_name}' is not supported. choose from: {self.SUPPORTED_MODELS}")

        # init featurizer
        featurizer = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        embed_dim = featurizer.num_features
        
        # init classifier head
        classifier_head = nn.Linear(embed_dim, num_classes)
        
        super().__init__(featurizer=featurizer, classifier_head=classifier_head, device=device)
        
        self.model_name = model_name
        self.embed_dim = embed_dim
        self.num_classes = num_classes