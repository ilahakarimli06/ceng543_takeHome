from data_loader import get_dataloaders
from model import BiLSTM, BiGRU, BERTBiLSTM, BERTBiGRU
from train import train, evaluate
from utils import set_seed
import torch
import json
from datetime import datetime
import os


def save_model_results(model, model_name, train_loader, val_loader, test_loader, device, mode, results_dict, output_dir):
    """Train, evaluate, and save a model's results."""
    print(f"\nTraining {model_name}...")
    history = train(model, train_loader, val_loader, device, mode=mode)
    
    # Save model weights
    model_filename = os.path.join(output_dir, f"{model_name.lower().replace(' ', '_').replace('+', '')}_model.pt")
    torch.save(model.state_dict(), model_filename)
    print(f"✓ Saved best model to: {model_filename}")
    
    # Evaluate on test set (ensure model is in eval mode)
    model.eval()
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


def main():
    # Set seed for reproducibility
    set_seed(42)
    
    # Create output directory if it doesn't exist
    output_dir = "question1/output"
    os.makedirs(output_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hidden_dim = 128
    num_classes = 2
    
    # Dictionary to store all results
    results = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "device": str(device),
        "models": {}
    }
    
    # ----------------------------------------
    # STATIC EMBEDDINGS (GloVe) + BiLSTM/BiGRU
    # ----------------------------------------
    print("\n" + "="*70)
    print("STATIC EMBEDDINGS (GloVe)")
    print("="*70)
    
    train_loader, val_loader, test_loader, meta = get_dataloaders(
        mode="static",
        limit_train=25000,
        limit_test=5000
    )
    embedding_dim = meta["embedding_dim"]
    
    print("\n[1/4] GloVe + BiLSTM")
    lstm_model = BiLSTM(embedding_dim, hidden_dim, num_classes)
    save_model_results(lstm_model, "GloVe + BiLSTM", train_loader, val_loader, test_loader, device, "static", results["models"], output_dir)

    print("\n" + "-"*70)
    print("\n[2/4] GloVe + BiGRU")
    gru_model = BiGRU(embedding_dim, hidden_dim, num_classes)
    save_model_results(gru_model, "GloVe + BiGRU", train_loader, val_loader, test_loader, device, "static", results["models"], output_dir)
    
    # --------------------------------------------
    # CONTEXTUAL EMBEDDINGS (BERT) + BiLSTM/BiGRU
    # --------------------------------------------
    print("\n" + "-"*70)
    print("CONTEXTUAL EMBEDDINGS (BERT)")
    print("-"*70)
    
    train_loader, val_loader, test_loader, meta = get_dataloaders(
        mode="contextual",
        limit_train=25000,
        limit_test=5000
    )
    bert_model_name = meta["model_name"]
    
    print("\n[3/4] BERT + BiLSTM")
    bert_lstm_model = BERTBiLSTM(bert_model_name, hidden_dim, num_classes)
    save_model_results(bert_lstm_model, "BERT + BiLSTM", train_loader, val_loader, test_loader, device, "contextual", results["models"], output_dir)

    print("\n" + "-"*70)
    print("\n[4/4] BERT + BiGRU")
    bert_gru_model = BERTBiGRU(bert_model_name, hidden_dim, num_classes)
    save_model_results(bert_gru_model, "BERT + BiGRU", train_loader, val_loader, test_loader, device, "contextual", results["models"], output_dir)
    
    # Save all results to JSON file
    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)
    
    print("\n" + "="*70)
    print("ALL COMBINATIONS COMPLETED")
    print(f"Results saved to: {results_path}")
    print("="*70)

if __name__ == "__main__":
    main()
