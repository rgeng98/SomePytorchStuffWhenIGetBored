import gengAI  # Assuming this is your custom module
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

if __name__ == "__main__":
    input_dim = [1024, 1024]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load datasets
    train = gengAI.Loader.StarsGoalHornDataset("starsGoalDatabase/train", input_dim)
    val = gengAI.Loader.StarsGoalHornDataset("starsGoalDatabase/val", input_dim)
    train_dataloader = DataLoader(train, batch_size=8, shuffle=True)
    val_dataloader = DataLoader(val, batch_size=8, shuffle=False) 

    myNN = gengAI.NN.Convolutional.CNN(input_dim, 7, 7, n_outputs=1).to(device)
    print(myNN)

    # Loss and optimizer
    criterion = torch.nn.BCEWithLogitsLoss().to(device)
    optimizer = torch.optim.Adam(myNN.parameters(), lr=0.0001) 

    loss_history = []
    val_loss_history = []
    val_acc_history = []
    epochs = 25 

    print(f"The neural network has {count_parameters(myNN):,} parameters")

    for e in range(epochs):
        print(f"Epoch: {e+1} / {epochs}")
        
        myNN.train()
        epoch_loss = 0
        scale = 0
        for inputs, targets in train_dataloader:
            inputs, targets = inputs.to(device), targets.to(device).float()  # Ensure float
            optimizer.zero_grad()
            outputs = myNN(inputs)  # [batch_size, 1]
            loss = criterion(outputs.squeeze(), targets)  # Squeeze to [batch_size]
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            scale += 1

        myNN.eval()
        epoch_val_loss = 0
        vscale = 0
        val_accuracy = 0
        with torch.no_grad():
            for inputs, targets in val_dataloader:
                inputs, targets = inputs.to(device), targets.to(device).float()
                outputs = myNN(inputs)
                loss = criterion(outputs.squeeze(), targets)
                epoch_val_loss += loss.item()
                # Calculate accuracy
                preds = (torch.sigmoid(outputs.squeeze()) > 0.5).float()
                val_accuracy += (preds.eq(targets).sum().item() / targets.size(0))
                vscale += 1

        train_loss = epoch_loss / scale
        val_loss = epoch_val_loss / vscale
        val_acc = val_accuracy / vscale
        print(f"Training loss:   {train_loss:.4f}")
        print(f"Validation loss: {val_loss:.4f}")
        print(f"Validation acc:  {val_acc:.4f}")
        loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        val_acc_history.append(val_acc)

    # Plot loss
    plt.figure()
    plt.plot(loss_history, label="Training Loss")
    plt.plot(val_loss_history, label="Validation Loss")
    plt.plot(val_acc_history, label="Validation Accuracy")
    plt.plot()
    plt.legend()
    plt.savefig("LossHistory.png")

    torch.save(myNN, "StarsGoalHornAI.pth")
    print("Model saved to StarsGoalHornAI.pth")