import copy
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from config import *

def train(model, train_loader, val_loader, criterion, optimizer, device):
    patience = 30
    best_val_loss = float("inf")
    patience_counter = 0
    best_weights = None
    early_stopping = False

    # liste per i plot
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(Config.MAX_EPOCHS):
        model.train()
        running_loss, correct, total = 0, 0, 0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{Config.MAX_EPOCHS}"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            outputs = outputs['clipwise_output']
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # ---- VALIDATION ----
        model.eval()
        val_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                outputs = outputs['clipwise_output']
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        val_loss = val_loss / total
        val_acc = correct / total
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # loss and accuracy plots during epochs
        if epoch % 10 == 0 and epoch != 0:
            print_loss(train_losses, val_losses, train_accs, val_accs)

        print(f"Epoch {epoch + 1}/{Config.MAX_EPOCHS} | "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # ---- EARLY STOPPING ----
        if early_stopping is True:
            if val_loss < best_val_loss:
               best_val_loss = val_loss
               patience_counter = 0
               best_weights = copy.deepcopy(model.state_dict())
            else:
               patience_counter += 1
               print(f"Patience_counter: {patience_counter}")
               if patience_counter >= patience:
                   print(f"Early stopping at epoch {epoch + 1}")
                   if best_weights is not None:
                       model.load_state_dict(best_weights)
                   break

        print_loss(train_losses, val_losses, train_accs, val_accs)

    return model, (train_losses, val_losses, train_accs, val_accs)

def print_loss(train_losses, val_losses, train_accs, val_accs, unlearning = False):
    if not unlearning:
        label1, label2, label3, label4 = "Train Loss", "Val Loss", "Train Acc", "Val Acc"
    else:
        label1, label2, label3, label4 = "Forget Loss", "Retain Loss", "Forget Acc", "Retain Acc"

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label=label1)
    plt.plot(val_losses, label=label2)
    plt.legend()
    plt.title("Loss")
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label=label3)
    plt.plot(val_accs, label=label4)
    plt.legend()
    plt.title("Accuracy")

    if not unlearning:
        plt.savefig(f"results/{Config.name_path()}_LOSS.png", bbox_inches='tight')
    else:
        plt.savefig(f"results/{Config.UNL_NAME}_LOSS.png", bbox_inches='tight')
    plt.show()