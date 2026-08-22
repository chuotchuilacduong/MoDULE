import torch

def test_acc(model, test_loader, device="cuda"):
    """
    calculates the accuracy of the model on the test_set.
    used to measure if the unlearned model still generalizes well to unseen data.
    
    args:
        model: architecture inherited from BaseArchitecture.
        test_loader: DataLoader containing the unseen test data.
        device: "cuda" or "cpu".
        
    returns:
        float: the accuracy as a ratio (between 0.0 and 1.0).
    """
    correct = 0
    total = 0
    
    for batch in test_loader:
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