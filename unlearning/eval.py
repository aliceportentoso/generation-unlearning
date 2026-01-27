import numpy
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from train import *

def evaluate(model, data_loader, label_encoder):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(inputs)
            outputs = outputs['clipwise_output']
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = compute_accuracy(model, data_loader)
    print(f"Accuracy: {acc:.4f}")

    print_confusion_matrix(all_labels, all_preds, label_encoder)
    plt.title(f"Normalized confusion Matrix for LEARNING of {Config.SUBSET} subset, {Config.UNL_EPOCHS} unl epochs")

    if Config.GENRE_TO_REMOVE is None:
        plt.title(f"Normalized confusion Matrix for GENRE CLASSIFICATION, {Config.MAX_EPOCHS} epochs")
        plt.savefig(f"results/{Config.UNL_NAME}_CM.png", bbox_inches='tight')
    else:
        plt.title(f"Normalized confusion Matrix for LEARNING WITHOUT {Config.GENRE_TO_REMOVE}, {Config.MAX_EPOCHS} epochs")
        plt.savefig(f"results/{Config.name_path}_CM.png", bbox_inches='tight')

    return acc

def evaluate_unlearning(model, forget_loader, retain_loader, val_loader, label_encoder):

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(inputs)
            outputs = outputs['clipwise_output']
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    forget_acc = compute_accuracy(model, forget_loader)
    print(f"Accuracy sui dati da dimenticare: {forget_acc:.4f}")
    retain_acc = compute_accuracy(model, retain_loader)
    print(f"Accuracy sui dati rimasti: {retain_acc:.4f}")
    #global_acc = compute_accuracy(model, val_loader)
    #print(f"Accuracy sui dati totale: {global_acc:.4f}")

    print_confusion_matrix(all_labels, all_preds, label_encoder)
    plt.title(f"Normalized confusion Matrix for UNLEARNING of {Config.GENRE_TO_FORGET}, {Config.UNL_EPOCHS} unl epochs")
    plt.savefig(f"results/{Config.UNL_NAME}_CM.png", bbox_inches='tight')

    return forget_acc, retain_acc

def compute_accuracy(model, loader):
    model.to(DEVICE)
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(inputs)
            outputs = outputs['clipwise_output']
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    acc = correct / total
    return acc

def print_confusion_matrix(all_labels, all_preds, label_encoder):
    cm = confusion_matrix(all_labels, all_preds, labels=range(Config.NUM_CLASSES))
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, numpy.newaxis] * 100
    fig, ax = plt.subplots(figsize=(10, 10))
    disp = ConfusionMatrixDisplay(cm_percentage, display_labels=label_encoder.classes_)
    disp.plot(ax=ax, cmap=plt.get_cmap('Blues'), colorbar=False)
    plt.xticks(rotation=90)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')