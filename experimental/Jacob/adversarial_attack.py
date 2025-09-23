import torch
import torch.nn as nn
from dataset import PendulumDataset
from torch.utils.data import DataLoader
from models import SimpleNN
from sklearn.preprocessing import StandardScaler
import numpy as np

def pgd_attack_regression(model, inputs, targets, eps=0.3, alpha=0.01, iters=40):
    """
    PGD attack for regression models.
    Args:
        model: regression model
        inputs: input tensor (features)
        targets: target tensor (regression targets)
        eps: maximum perturbation
        alpha: step size
        iters: number of iterations
    Returns:
        adversarial examples
    """
    inputs = inputs.clone().detach().to(inputs.device).type(torch.float32)
    targets = targets.to(inputs.device).type(torch.float32)
    loss_fn = nn.MSELoss()
    ori_inputs = inputs.data

    for i in range(iters):
        inputs.requires_grad = True
        outputs = model(inputs)
        model.zero_grad()
        loss = loss_fn(outputs, targets)
        loss.backward()
        adv_inputs = inputs + alpha * inputs.grad.sign()
        eta = torch.clamp(adv_inputs - ori_inputs, min=-eps, max=eps)
        inputs = torch.clamp(ori_inputs + eta, min=inputs.min(), max=inputs.max()).detach()

    return inputs


def evaluate_model(model, data_loader, device):
    model.eval()
    total_loss = 0.0
    loss_fn = nn.MSELoss()
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
    avg_loss = total_loss / len(data_loader.dataset)
    return avg_loss


if __name__ == "__main__":
    # Load dataset
    print("Loading dataset...")
    dataset = PendulumDataset('data/pendulum_simulation.csv')
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    # Normalize dataset

    X = dataset.X
    y = dataset.y

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_norm = scaler_X.fit_transform(X)
    y_norm = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

    # Replace dataset with normalized values
    dataset.X = torch.tensor(X_norm, dtype=torch.float32)
    dataset.y = torch.tensor(y_norm, dtype=torch.float32)

    # Load pre-trained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleNN(input_dim=dataset.X.shape[1], hidden_dim=256, output_dim=1)
    model.load_state_dict(torch.load("data/pendulum_model.pth", weights_only=True))
    model.to(device)

    # Evaluate clean performance
    clean_loss = evaluate_model(model, data_loader, device)
    print(f"Clean MSE Loss: {clean_loss:.6f}")

    # Perform PGD attack
    adv_examples = []
    for inputs, targets in data_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        adv_inputs = pgd_attack_regression(model, inputs, targets, eps=0.1, alpha=0.01, iters=40)
        adv_examples.append((adv_inputs.cpu(), targets.cpu()))
    
    print(f"Generated {len(adv_examples)} adversarial examples.")

    # Display the first three clean examples next to their adversarial counterparts and their losses
    print("\nFirst 3 clean vs adversarial examples with losses:")
    loss_fn = nn.MSELoss()
    model.eval()
    for i in range(min(3, len(adv_examples))):
        clean_x, clean_y = dataset[i]
        adv_x, adv_y = adv_examples[i]

        clean_x = clean_x.unsqueeze(0).to(device)
        clean_y = clean_y.unsqueeze(0).to(device)
        adv_x = adv_x.to(device)
        adv_y = adv_y.to(device)

        with torch.no_grad():
            clean_pred = model(clean_x)
            adv_pred = model(adv_x)
            clean_loss = loss_fn(clean_pred, clean_y).item()
            adv_loss = loss_fn(adv_pred, adv_y).item()

        print(f"Clean Input: {clean_x.cpu().squeeze().tolist()}, Target: {clean_y.cpu().squeeze().tolist()}, Loss: {clean_loss:.6f}")
        print(f"Adv  Input: {adv_x.cpu().squeeze().tolist()}, Target: {adv_y.cpu().squeeze().tolist()}, Loss: {adv_loss:.6f}")
        print("-" * 40)

    # Evaluate adversarial performance
    adv_dataset = torch.utils.data.TensorDataset(torch.cat([x[0] for x in adv_examples]), torch.cat([x[1] for x in adv_examples]))
    adv_loader = DataLoader(adv_dataset, batch_size=64, shuffle=False)
    adv_loss = evaluate_model(model, adv_loader, device)
    print(f"Adversarial MSE Loss: {adv_loss:.6f}")