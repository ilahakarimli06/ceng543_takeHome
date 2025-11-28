from data_loader import get_dataloaders
from model import BERTBiLSTM
from train import train, evaluate
from utils import set_seed
import torch
import json
from datetime import datetime
import builtins
import re
from contextlib import contextmanager


def save_model_results(model, model_name, train_loader, val_loader, test_loader, device, mode, results_dict):
    """Train, evaluate, and save a model's results."""
    num_epochs = 10
    print(f"\nTraining {model_name}...")
    with epoch_progress_monitor(num_epochs):
        history = train(model, train_loader, val_loader, device, mode=mode, num_epochs=num_epochs)
    print(f"[100.0%] Training completed ({num_epochs}/{num_epochs} epochs)")
    
    # Save model weights
    model_filename = f"{model_name.lower().replace(' ', '_').replace('+', '')}_model.pt"
    torch.save(model.state_dict(), model_filename)
    
    # Evaluate on test set
    print(f"\nTest Results for {model_name}:")
    test_metrics = evaluate(model, test_loader, device, mode=mode)
    print(f"Loss: {test_metrics['loss']:.4f} | Accuracy: {test_metrics['accuracy']:.4f} | F1: {test_metrics['macro_f1']:.4f}")
    
    # Store results
    results_dict[model_name.replace(' ', '_').replace('+', '')] = {
        "test_loss": test_metrics['loss'],
        "test_accuracy": test_metrics['accuracy'],
        "test_macro_f1": test_metrics['macro_f1'],
        "training_history": {
            "train_loss": history.train_loss,
            "val_acc": history.val_acc,
            "val_macro_f1": history.val_macro_f1,
            "epoch_time_sec": history.epoch_time_sec
        }
    }


@contextmanager
def epoch_progress_monitor(total_epochs: int):
    """Temporarily patch print to prepend epoch progress percentage."""
    original_print = builtins.print
    epoch_pattern = re.compile(r"Epoch\s+(\d+)/(\d+)")

    def progress_print(*args, **kwargs):
        if args and isinstance(args[0], str):
            text = args[0]
            match = epoch_pattern.match(text)
            if match:
                current = int(match.group(1))
                percent = (current / total_epochs) * 100 if total_epochs else 0.0
                text = f"[{percent:5.1f}%] {text}"
                args = (text,) + args[1:]
        return original_print(*args, **kwargs)

    original_print(f"[  0.0%] Training started (0/{total_epochs} epochs)")
    builtins.print = progress_print
    try:
        yield
    finally:
        builtins.print = original_print


def main():
    # Set seed for reproducibility
    set_seed(42)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hidden_dim = 128
    num_classes = 2
    
    # Dictionary to store all results
    results = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "device": str(device),
        "models": {}
    }
    
    #############################################
    # CONTEXTUAL EMBEDDINGS (BERT) + BiLSTM
    #############################################
    print("\n" + "-"*70)
    print("CONTEXTUAL EMBEDDINGS (BERT)")
    print("-"*70)
    
    train_loader, val_loader, test_loader, meta = get_dataloaders(
        mode="contextual",
        limit_train=25000,
        limit_test=5000,
        max_len=256
    )
    bert_model_name = meta["model_name"]
    
    print("\nBERT + BiLSTM")
    bert_lstm_model = BERTBiLSTM(bert_model_name, hidden_dim, num_classes)
    save_model_results(bert_lstm_model, "BERT + BiLSTM", train_loader, val_loader, test_loader, device, "contextual", results["models"])
    
    # Save results to JSON file
    with open("bert_bilstm_results.json", "w") as f:
        json.dump(results, f, indent=4)
    
    print("\n" + "="*70)
    print("BERT + BiLSTM TRAINING COMPLETED")
    print("Results saved to: bert_bilstm_results.json")
    print("="*70)

if __name__ == "__main__":
    main()

