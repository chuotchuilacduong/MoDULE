import torch

def retain_acc(model, retain_loader, device="cuda"):
    """
    calculates the accuracy of the model on the retain_set.
    used to measure if the model has maintained its knowledge on the data it should keep.
    
    args:
        model: architecture inherited from BaseArchitecture.
        retain_loader: DataLoader containing the retain data.
        device: "cuda" or "cpu".
        
    returns:
        float: the accuracy as a ratio (between 0.0 and 1.0).
    """
    correct = 0
    total = 0
    
    for batch in retain_loader:
        # safely extracts images and true labels, ignoring domains if present
        images = batch[0].to(device)
        labels = batch[1].to(device)
        
        # use inference mode (sets eval() and disables gradients)
        logits, _ = model.inference(images)
        
        # get predictions and compare to ground truth
        predictions = torch.argmax(logits, dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
        
    accuracy = correct / total
    return accuracy