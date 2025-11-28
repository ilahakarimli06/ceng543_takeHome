import torch
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from tqdm import tqdm
from utils import set_seed, EpochTimer, prepare_batch, History


def train(model, train_loader, val_loader, device, mode="static", num_epochs=10, patience=3, lr=1e-3, seed=42):
    # Set seed for reproducibility
    set_seed(seed)
    
    criterion = torch.nn.CrossEntropyLoss()
    if mode == "contextual":
        # Use different learning rates for BERT and RNN layers
        param_groups = []
        if hasattr(model, "bert"):
            param_groups.append({
                "params": model.bert.parameters(),
                "lr": 1e-5,
                "weight_decay": 0.01,
            })
        rnn_params = []
        if hasattr(model, "lstm"):
            rnn_params.extend(model.lstm.parameters())
        if hasattr(model, "gru"):
            rnn_params.extend(model.gru.parameters())
        if rnn_params:
            param_groups.append({
                "params": rnn_params,
                "lr": lr,
                "weight_decay": 0.01,
            })
        if hasattr(model, "fc"):
            param_groups.append({
                "params": model.fc.parameters(),
                "lr": lr,
                "weight_decay": 0.01,
            })
        optimizer = torch.optim.AdamW(param_groups)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    model.to(device)
    
    # Initialize history to track metrics
    history = History(
        train_loss=[],
        val_acc=[],
        val_macro_f1=[],
        epoch_time_sec=[]
    )
    
    best_val_f1 = 0.0
    best_model_state = None
    patience_counter = 0

    for epoch in range(num_epochs):
        # Measure epoch time for convergence efficiency
        with EpochTimer() as timer:
            model.train()
            train_losses = []
            all_preds, all_labels = [], []

            # Progress bar for batches
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", 
                       unit="batch", leave=False, ncols=100)
            
            for batch in pbar:
                inputs, labels = prepare_batch(batch, device, mode)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()

                # Prevent gradient explosion
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                train_losses.append(loss.item())
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                # Update progress bar with current loss
                pbar.set_postfix({'loss': f'{loss.item():.4f}', 
                                 'avg_loss': f'{np.mean(train_losses):.4f}'})

            train_acc = accuracy_score(all_labels, all_preds)
            train_f1 = f1_score(all_labels, all_preds, average="macro")

            # --- Validation ---
            val_metrics = evaluate(model, val_loader, device, mode=mode)

        # Store metrics in history
        history.add(
            loss=np.mean(train_losses),
            acc=val_metrics['accuracy'],
            f1=val_metrics['macro_f1'],
            tsec=timer.seconds
        )
        
        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {np.mean(train_losses):.4f} | "
              f"Val Loss: {val_metrics['loss']:.4f} | "
              f"Val Acc: {val_metrics['accuracy']:.4f} | Val F1: {val_metrics['macro_f1']:.4f} | "
              f"Time: {timer.seconds:.2f}s")

        # --- Early stopping with best model checkpoint ---
        if val_metrics['macro_f1'] > best_val_f1:
            best_val_f1 = val_metrics['macro_f1']
            # Save best model state (deep copy to CPU to avoid GPU memory issues)
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            print(f"New best model (Val F1: {best_val_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    # Load best model state before returning
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        model.to(device)
        print(f"Loaded best model checkpoint (Val F1: {best_val_f1:.4f})")
    
    print(f"Best Validation F1: {best_val_f1:.4f}")
    
    # Calculate convergence efficiency metrics
    if history.epoch_time_sec:
        avg_epoch_time = np.mean(history.epoch_time_sec)
        total_time = sum(history.epoch_time_sec)
        print(f"Convergence Efficiency: Avg epoch time: {avg_epoch_time:.2f}s, Total time: {total_time:.2f}s")
    
    return history



def evaluate(model, data_loader, device, mode="static"):
    
    #Computes average loss, accuracy, and macro F1 score.
    
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        pbar = tqdm(data_loader, desc="Validating", unit="batch", leave=False, ncols=100)
        for batch in pbar:
            inputs, labels = prepare_batch(batch, device, mode)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(data_loader)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")

    return {
        'loss': avg_loss,
        'accuracy': acc,
        'macro_f1': f1
    }