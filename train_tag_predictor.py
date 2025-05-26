import copy
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import torch.optim as optim

def train_tag_predictor(
    img_dir,
    csv_file,
    label_cols,
    train_loader,
    val_loader,
    metrics_history,
    alpha,
    epochs=5,
    batch_size=64,
    lr=1e-4,
    device='cuda',
    early_stopping_patience=3  # new
):
    model = build_model(num_classes=len(label_cols)).to(device)
    criterion = FocalLoss(alpha=alpha, gamma=5)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    best_val_auc = 0.0
    best_thresholds = None
    best_model_wts = copy.deepcopy(model.state_dict())
    epochs_since_improvement = 0  # new

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        all_labels = []
        all_preds = []

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Training]"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            all_labels.append(labels.detach().cpu())
            all_preds.append(torch.sigmoid(outputs).detach().cpu())

        train_loss = running_loss / len(train_loader.dataset)
        y_train_true = torch.cat(all_labels).numpy()
        y_train_pred = torch.cat(all_preds).numpy()

        # Train AUC
        aucs = []
        for i in range(len(label_cols)):
            try:
                auc = roc_auc_score(y_train_true[:, i], y_train_pred[:, i])
            except ValueError:
                auc = float('nan')
            aucs.append(auc)
        mean_train_auc = np.nanmean(aucs)

        # Validation
        model.eval()
        val_labels = []
        val_preds = []
        val_running_loss = 0.0

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [Validation]"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item() * images.size(0)
                val_labels.append(labels.cpu())
                val_preds.append(torch.sigmoid(outputs).cpu())

        val_loss = val_running_loss / len(val_loader.dataset)
        y_val_true = torch.cat(val_labels).numpy()
        y_val_pred = torch.cat(val_preds).numpy()

        # Thresholds + AUC
        current_thresholds = find_best_thresholds(y_val_true, y_val_pred)
        y_pred_bin = apply_thresholds(y_val_pred, current_thresholds)

        val_aucs = []
        for i in range(len(label_cols)):
            try:
                auc = roc_auc_score(y_val_true[:, i], y_val_pred[:, i])
            except ValueError:
                auc = float('nan')
            val_aucs.append(auc)
        mean_val_auc = np.nanmean(val_aucs)

        val_metrics = evaluate_model(y_val_true, y_val_pred, current_thresholds)

        print(f"\nEpoch {epoch}/{epochs}")
        print(f"Train Loss: {train_loss:.4f} | Train Mean AUC: {mean_train_auc:.4f} | Val Loss: {val_loss:.4f} | Val Mean AUC: {mean_val_auc:.4f}")
        for label, metrics in val_metrics.items():
            print(f"\n{label} : {metrics}")

        # Early stopping logic
        if mean_val_auc > best_val_auc:
            print("Validation AUC improved. Saving model and thresholds.")
            best_val_auc = mean_val_auc
            best_thresholds = current_thresholds
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1
            print(f"No improvement in Val AUC for {epochs_since_improvement} epoch(s).")
            if epochs_since_improvement >= early_stopping_patience:
                print("Early stopping triggered.")
                break

    model.load_state_dict(best_model_wts)
    print(f"\nBest Val AUC: {best_val_auc:.4f} — thresholds restored from best epoch.")

    return model, best_thresholds
