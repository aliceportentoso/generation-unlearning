import joblib
from eval import evaluate_unlearning
from dataset import FMADataset
from model import Cnn6
import time
from torch.utils.data import DataLoader
from config import *
import torch
from train import print_loss
from stable_audio_tools import get_pretrained_model


def unlearning_main():
    retain_ids, forget_ids, retain_labels, forget_labels = [], [], [], []
    start_time = time.time()

    # --- CARICA MODELLO E LABEL ENCODER ---

    model, model_config = get_pretrained_model("stabilityai/stable-audio-open-1.0")
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=Config.DEVICE)) # carica i pesi salvati dall'addestramento

    model.eval()
    le = joblib.load(Config.ENCODER_PATH)

    # carica gli split
    dir_ = f"data_splits/{Config.SUBSET}-dataset_remove-None"
    train_ids = joblib.load(f"{dir_}/train_ids.joblib")
    train_labels = joblib.load(f"{dir_}/train_labels.joblib")
    val_ids = joblib.load(f"{dir_}/val_ids.joblib")
    val_labels = joblib.load(f"{dir_}/val_labels.joblib")

    retain_dataset = FMADataset(retain_ids, retain_labels)
    forget_dataset = FMADataset(forget_ids, forget_labels)
    val_dataset = FMADataset(val_ids, val_labels)

    retain_loader = DataLoader(retain_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    forget_loader = DataLoader(forget_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LR)

    # --- ALGORITMI DI UNLEARNING ---
    if Config.UNL_METHOD == "FT":
        unl_fine_tuning(model, forget_loader, retain_loader, val_loader, criterion, optimizer, le, lambda_unlearn=l)
    elif Config.UNL_METHOD == "GA":
        unl_gradient_ascent(model, forget_loader, retain_loader, val_loader, criterion, optimizer, le, alpha=a, beta=b)
    elif Config.UNL_METHOD == "ST":
        unl_stochastic_teacher(model, forget_loader, retain_loader, val_loader, criterion, optimizer, le, alpha=a, beta=b) #0.3,0.7
    elif Config.UNL_METHOD == "OSM":
        unl_one_shot_magnitude(model, forget_loader, retain_loader, val_loader, criterion, optimizer, le, prune_ratio=0.5, ft_epochs=2)
    elif Config.UNL_METHOD == "A":
        unl_amnesiac(model, forget_loader, retain_loader, val_loader, criterion, le)
    else:
        print("unknown method")

    print(f"Tempo Unlearning: {(time.time() - start_time)/3600:.2f} ore")

def unl_fine_tuning(model, forget_loader, retain_loader, val_loader, criterion, optimizer, le, lambda_unlearn):
    print(f"LAMBDA UNLEARN: {lambda_unlearn}")
    intermedie = False

    model.train()
    forget_losses, retain_losses = [], []
    forget_accs, retain_accs = [], []
    retain_iter = iter(retain_loader)

    for epoch in range(Config.UNL_EPOCHS):
        total_forget, total_retain = 0.0, 0.0
        num_batches = 0

        for batch_forget in forget_loader:
            try:
                batch_retain = next(retain_iter)
            except StopIteration:
                retain_iter = iter(retain_loader)
                batch_retain = next(retain_iter)

            # --- Forget batch ---
            inputs_forget, labels_forget = [d.to(Config.DEVICE) for d in batch_forget]
            outputs_forget = model(inputs_forget)
            if isinstance(outputs_forget, dict) and "clipwise_output" in outputs_forget:
                outputs_forget = outputs_forget["clipwise_output"]
            forget_loss = criterion(outputs_forget, labels_forget)

            # --- Retain batch ---
            inputs_retain, labels_retain = [d.to(Config.DEVICE) for d in batch_retain]
            outputs_retain = model(inputs_retain)
            if isinstance(outputs_retain, dict) and "clipwise_output" in outputs_retain:
                outputs_retain = outputs_retain["clipwise_output"]
            retain_loss = criterion(outputs_retain, labels_retain)

            # --- Combined loss ---
            loss = retain_loss - lambda_unlearn * forget_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_forget += forget_loss.item()
            total_retain += retain_loss.item()
            num_batches += 1

        avg_forget = total_forget / num_batches
        avg_retain = total_retain / num_batches
        forget_losses.append(avg_forget)
        retain_losses.append(avg_retain)

        if intermedie == True:
            print(f"Epoch {epoch+1}/{Config.UNL_EPOCHS} | Retain: {avg_retain:.4f} | Forget: {avg_forget:.4f}")
            f_acc, r_acc = evaluate_unlearning(model, forget_loader, retain_loader, val_loader, le)
            forget_accs.append(f_acc)
            retain_accs.append(r_acc)
            print_loss(forget_losses, retain_losses, forget_accs, retain_accs, unlearning=True)

        if (epoch + 1) % 2 == 0:
            print(f"Epoch {epoch+1}:")
            evaluate_unlearning(model, forget_loader, retain_loader, val_loader, le)
    return

def unl_gradient_ascent(model, forget_loader, retain_loader, val_loader, criterion, optimizer, le, alpha, beta):
    print(f"ALPHA: {alpha}")
    print(f"BETA: {beta}")
    intermedie = False

    model.train()
    forget_losses, retain_losses = [], []
    forget_accs, retain_accs = [], []
    total_forget, total_retain = 0.0, 0.0
    num_batches = 0

    for epoch in range(Config.UNL_EPOCHS):
        forget_iter = iter(forget_loader)
        retain_iter = iter(retain_loader)

        for _ in range(min(len(forget_loader), len(retain_loader))):
            try:
                inputs_forget, labels_forget = next(forget_iter)
                inputs_forget, labels_forget = inputs_forget.to(Config.DEVICE), labels_forget.to(Config.DEVICE)
            except StopIteration:
                break

            try:
                inputs_retain, labels_retain = next(retain_iter)
                inputs_retain, labels_retain = inputs_retain.to(Config.DEVICE), labels_retain.to(Config.DEVICE)
            except StopIteration:
                break

            optimizer.zero_grad()

            # --- Forget loss (Gradient Ascent) ---
            outputs_forget = model(inputs_forget)['clipwise_output']
            forget_loss = criterion(outputs_forget, labels_forget)

            # --- Retain loss (Gradient Descent) ---
            outputs_retain = model(inputs_retain)['clipwise_output']
            retain_loss = criterion(outputs_retain, labels_retain)

            total_forget += forget_loss.item()
            total_retain += retain_loss.item()
            num_batches += 1

            loss = -alpha * forget_loss + beta * retain_loss
            loss.backward()
            optimizer.step()

        avg_forget = total_forget / num_batches
        avg_retain = total_retain / num_batches
        forget_losses.append(avg_forget)
        retain_losses.append(avg_retain)

        if intermedie is True:
            print(f"Epoch {epoch + 1}/{Config.UNL_EPOCHS} | Retain: {avg_retain:.4f} | Forget: {avg_forget:.4f}")
            f_acc, r_acc = evaluate_unlearning(model, forget_loader, retain_loader, val_loader, le)
            forget_accs.append(f_acc)
            retain_accs.append(r_acc)
            print_loss(forget_losses, retain_losses, forget_accs, retain_accs, unlearning=True)

        if (epoch + 1) % 2 == 0:
            print(f"Epoch {epoch+1}:")
            evaluate_unlearning(model, forget_loader, retain_loader, val_loader, le)
    return

def unl_stochastic_teacher(model, forget_loader, retain_loader, val_loader, criterion, optimizer, le, alpha=0.3, beta=0.7, randomize_labels=False):
    """   randomize_labels: se True, randomizza le label dei dati da dimenticare  """
    print(f'alpha: {alpha}')
    print(f'beta: {beta}')
    model.train()
    model.to(Config.DEVICE)

    forget_losses, retain_losses = [], []
    forget_accs, retain_accs = [], []

    for epoch in range(Config.UNL_EPOCHS):
        retain_iter = iter(retain_loader)
        forget_iter = iter(forget_loader)

        num_batches = min(len(retain_iter), len(forget_iter))

        for _ in range(num_batches):
            try:
                x_retain, y_retain = next(retain_iter)
                x_forget, y_forget = next(forget_iter)
            except StopIteration:
                break

            x_retain, y_retain = x_retain.to(Config.DEVICE), y_retain.to(Config.DEVICE)
            x_forget, y_forget = x_forget.to(Config.DEVICE), y_forget.to(Config.DEVICE)

            optimizer.zero_grad()

            # Forward pass retain e forget
            out_retain = model(x_retain)['clipwise_output']
            retain_loss = criterion(out_retain, y_retain)
            out_forget = model(x_forget)['clipwise_output']

            if randomize_labels:
                # Random teacher: mescola o randomizza le etichette
                y_rand = torch.randint_like(y_forget, low=0, high=out_forget.size(1))
                forget_loss = -criterion(out_forget, y_rand)  # loss negativa = dimenticare
            else:
                # Alternativamente, incoraggia uniformità (incertezza)
                probs = torch.nn.functional.log_softmax(out_forget, dim=1)
                uniform = torch.full_like(probs, 1.0 / probs.size(1))
                forget_loss = torch.nn.functional.kl_div(probs, uniform, reduction='batchmean')

            loss = alpha * retain_loss + beta * forget_loss
            loss.backward()
            optimizer.step()

        forget_losses.append(forget_loss)
        retain_losses.append(retain_loss)

        print(f"Epoch {epoch + 1}/{Config.UNL_EPOCHS} | Retain: {retain_loss:.4f} | Forget: {forget_loss:.4f}")
        f_acc, r_acc = evaluate_unlearning(model, forget_loader, retain_loader, val_loader, le)
        forget_accs.append(f_acc)
        retain_accs.append(r_acc)
        print_loss(forget_losses, retain_losses, forget_accs, retain_accs, unlearning=True)

    evaluate_unlearning(model, forget_loader, retain_loader, val_loader, le)
    print("STOCHASTICH TEACHER completato.")
    return forget_losses, retain_losses

def unl_one_shot_magnitude(model, forget_loader, retain_loader, val_loader, criterion, optimizer, le, prune_ratio, ft_epochs):

    parameters_to_prune = []
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
            parameters_to_prune.append((module, 'weight'))
            if hasattr(module, 'bias') and module.bias is not None:
                parameters_to_prune.append((module, 'bias'))

    # --- Step 1: Pruning ---
    torch.nn.utils.prune.global_unstructured(
        parameters_to_prune,
        pruning_method=torch.nn.utils.prune.L1Unstructured,
        amount=prune_ratio
    )

    # --- Step 2: Freeze pruned weights ---
    for module, param_name in parameters_to_prune:
        mask = getattr(module, f"{param_name}_mask")
        module.register_buffer(f"{param_name}_mask_blocked", mask.clone())
        param = getattr(module, param_name)
        param.grad = None
        param.register_hook(lambda grad, mask=mask: grad * mask)

    # --- Step 3: Retain fine-tuning ---
    model.train()
    for epoch in range(ft_epochs):
        for inputs_retain, labels_retain in retain_loader:
            inputs_retain, labels_retain = inputs_retain.to(Config.DEVICE), labels_retain.to(Config.DEVICE)
            outputs_retain = model(inputs_retain)['clipwise_output']
            retain_loss = criterion(outputs_retain, labels_retain)
            retain_loss.backward()
            optimizer.step()

    evaluate_unlearning(model, forget_loader, retain_loader, val_loader, le)
    return

def unl_amnesiac(model, forget_loader, retain_loader, val_loader, criterion, le):
    #             steps=2, lr_forget=3e-3, lr_retain=1e-3, lambda_stab=1e-4):
    # steps, lr_forget, lr_retain, lambda_stab
    model = model.to(Config.DEVICE)

    opt = torch.optim.SGD(model.parameters(), lr=1.0)  # LR scalato a mano

    retain_iter = iter(retain_loader)

    for step in range(steps):

        # === 1️⃣ get retain batch (per orthogonalization) ===
        try:
            x_r, y_r = next(retain_iter)
        except StopIteration:
            retain_iter = iter(retain_loader)
            x_r, y_r = next(retain_iter)

        x_r, y_r = x_r.to(Config.DEVICE), y_r.to(Config.DEVICE)

        opt.zero_grad()
        out_r = model(x_r)
        if isinstance(out_r, dict):
            out_r = out_r["clipwise_output"]
        retain_loss = criterion(out_r, y_r)

        retain_loss.backward()
        g_retain = []
        for p in model.parameters():
            if p.grad is None:
                g_retain.append(None)
            else:
                g_retain.append(p.grad.clone())

        # === 2️⃣ FORGET PHASE (with orthogonalized gradient ascent) ===
        for x_f, y_f in forget_loader:
            x_f, y_f = x_f.to(Config.DEVICE), y_f.to(Config.DEVICE)

            opt.zero_grad()
            out_f = model(x_f)
            if isinstance(out_f, dict):
                out_f = out_f["clipwise_output"]
            forget_loss = criterion(out_f, y_f)

            # ascent = -loss
            (-forget_loss).backward()

            # ---- AMNESIAC orthogonalization ----
            with torch.no_grad():
                for p, g_r in zip(model.parameters(), g_retain):
                    if p.grad is None:
                        continue
                    g_f = p.grad
                    if g_r is not None:  # evita crash
                        proj = (
                                       torch.dot(g_f.flatten(), g_r.flatten())
                                       / (g_r.norm() ** 2 + 1e-12)
                               ) * g_r
                        p.grad -= proj

            # manual LR for ascent
            with torch.no_grad():
                for p in model.parameters():
                    if p.grad is not None:
                        p -= lr_forget * p.grad

        # === 3️⃣ RETAIN PHASE (stabilized descent) ===
        opt.zero_grad()
        out_r = model(x_r)
        if isinstance(out_r, dict):
            out_r = out_r["clipwise_output"]

        retain_loss = criterion(out_r, y_r)

        # stabilization: keep parameters close to pre-unlearning weights
        l2_reg = sum((p**2).sum() for p in model.parameters())
        total_retain_loss = retain_loss + lambda_stab * l2_reg

        total_retain_loss.backward()

        with torch.no_grad():
            for p in model.parameters():
                if p.grad is not None:
                    p -= lr_retain * p.grad

    evaluate_unlearning(model, forget_loader, retain_loader, val_loader, le)

def forget_retain_split(train_ids, train_labels, le):

    idx_to_remove = le.transform([Config.GENRE_TO_FORGET])[0]
    print(f"Rimuovere il genere '{Config.GENRE_TO_FORGET}' (indice {idx_to_remove})")

    # Filtra i dati
    forget_ids, forget_labels, retain_ids, retain_labels = [], [], [], []

    for tid, label in zip(train_ids, train_labels):
        if label == idx_to_remove:
            forget_ids.append(tid)
            forget_labels.append(label)
        else:
            retain_ids.append(tid)
            retain_labels.append(label)

    return forget_ids, forget_labels, retain_ids, retain_labels

# START #################################

if Config.UNL_METHOD == "FT":
    lambdas = [0.3, 0.5, 0.7]
    lrs = [0.005, 0.0005, 0.00005]
    for l in lambdas:
        for lr in lrs:
            Config.LR = lr
            Config.UNL_NAME = f'FT_lambda-{l}_LR-{lr}_{Config.GENRE_TO_FORGET}_{Config.timestamp}'
            Config.print_config_unl()
            unlearning_main()

if Config.UNL_METHOD == "GA":
    alphas = [0.5, 1, 2]
    betas = [0.7, 1, 1.5]
    lrs = [0.005, 0.0005, 0.00005]
    for a in alphas:
        for b in betas:
            for lr in lrs:
                Config.LR = lr
                Config.UNL_NAME = f'GA_alpha-{a}_beta-{b}_LR-{lr}_{Config.GENRE_TO_FORGET}_{Config.timestamp}'
                Config.print_config_unl()
                unlearning_main()

########################################################

if Config.UNL_METHOD == "OSM":
    prune_ratio = 0.5
    ft_epochs = 2
    for genre in Config.GENRES:
        Config.GENRE_TO_FORGET = genre
        Config.unl_name_path()
        Config.print_config_unl()
        unlearning_main()

if Config.UNL_METHOD == "A":
    param_grid = {
        "steps": [2, 4],
        "lr_forget": [1e-3, 1e-2],
        "lr_retain": [1e-3, 1e-4],
        "lambda_stab": [1e-5, 1e-3]
    }

    # Esempio di ciclo su tutte le combinazioni possibili
    for steps in param_grid["steps"]:
        for lr_forget in param_grid["lr_forget"]:
            for lr_retain in param_grid["lr_retain"]:
                for lambda_stab in param_grid["lambda_stab"]:
                    print(f"Esecuzione con: steps={steps}, lr_forget={lr_forget}, "
                          f"lr_retain={lr_retain}, lambda_stab={lambda_stab}")

                    unlearning_main()

if Config.UNL_METHOD == "ST":
    a = 1
    b = 5
    for genre in Config.GENRES:
        Config.GENRE_TO_FORGET = genre
        Config.unl_name_path()
        Config.print_config_unl()
        unlearning_main()

else:
    Config.UNL_NAME = Config.unl_name_path()
    Config.print_config_unl()
    unlearning_main()