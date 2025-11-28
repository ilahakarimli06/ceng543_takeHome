"""
Train the best model configuration from ablation study
Configuration: 4 layers, 2 heads, random embedding
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import json
import time
from torch.utils.data import DataLoader

# Import from question3
from question3.models.transformer import TransformerModel
from question3.utils.embeddings import create_embedding_layer
from question2.utils.dataset import ParallelIdsDataset, collate_pad, PAD_ID
from question3.train_transformer import train_one_epoch, evaluate

import os

def train_best_config():
    """Train the best configuration: 4 layers, 2 heads, random embedding"""
    
    # Best config from ablation study
    NUM_LAYERS = 4
    NHEAD = 2
    D_MODEL = 256
    FEEDFORWARD_DIM = 512  # Ablation study uses 512
    VOCAB_SIZE = 8000
    BATCH_SIZE = 64
    NUM_EPOCHS = 20
    LEARNING_RATE = 3e-4  # Ablation study uses 3e-4 (0.0003)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"\nTraining Configuration:")
    print(f"  Layers: {NUM_LAYERS}")
    print(f"  Attention Heads: {NHEAD}")
    print(f"  Model Dimension: {D_MODEL}")
    print(f"  Embedding: Random")
    
    # Paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    TRAIN_SRC = os.path.join(base_dir, "question2/data/tokenized/train.de.ids")
    TRAIN_TGT = os.path.join(base_dir, "question2/data/tokenized/train.en.ids")
    VAL_SRC = os.path.join(base_dir, "question2/data/tokenized/validation.de.ids")
    VAL_TGT = os.path.join(base_dir, "question2/data/tokenized/validation.en.ids")
    SPM_MODEL = os.path.join(base_dir, "question2/data/spm_shared_unigram.model")
    
    # Create random embeddings
    print("\nCreating random embeddings...")
    source_emb = create_embedding_layer(VOCAB_SIZE, D_MODEL, "random", PAD_ID)
    target_emb = create_embedding_layer(VOCAB_SIZE, D_MODEL, "random", PAD_ID)
    
    # Data loaders
    print("Loading data...")
    train_loader = DataLoader(
        ParallelIdsDataset(TRAIN_SRC, TRAIN_TGT),
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda b: collate_pad(b, PAD_ID)
    )
    
    val_loader = DataLoader(
        ParallelIdsDataset(VAL_SRC, VAL_TGT),
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=lambda b: collate_pad(b, PAD_ID)
    )
    
    # Create model
    print("Creating Transformer model...")
    model = TransformerModel(
        source_vocab_size=VOCAB_SIZE,
        target_vocab_size=VOCAB_SIZE,
        model_dim=D_MODEL,
        num_heads=NHEAD,
        num_encoder_layers=NUM_LAYERS,
        num_decoder_layers=NUM_LAYERS,
        feedforward_dim=FEEDFORWARD_DIM,
        dropout=0.1,
        pad_id=PAD_ID,
        source_embedding=source_emb,
        target_embedding=target_emb
    )
    model.to(device)
    
    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=PAD_ID)
    
    # Training loop
    print(f"\nStarting training for {NUM_EPOCHS} epochs...")
    print("="*60)
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    start_time = time.time()
    
    for epoch in range(NUM_EPOCHS):
        epoch_start = time.time()
        
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        train_losses.append(train_loss)
        
        # Evaluate
        val_loss = evaluate(model, val_loader, device)
        val_losses.append(val_loss)
        
        epoch_time = time.time() - epoch_start
        
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val PPL: {torch.exp(torch.tensor(val_loss)):.2f} | "
              f"Time: {epoch_time:.1f}s")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(os.path.dirname(__file__), 'best_model_4l_2h.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'vocab_size': VOCAB_SIZE,
                'd_model': D_MODEL,
                'nhead': NHEAD,
                'num_layers': NUM_LAYERS,
                'feedforward_dim': FEEDFORWARD_DIM,
                'embedding_type': 'random'
            }, save_path)
            print(f"  → Best model saved! (val_loss: {val_loss:.4f})")
    
    total_time = time.time() - start_time
    print("\n" + "="*60)
    print(f"Training completed in {total_time/60:.1f} minutes")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best perplexity: {torch.exp(torch.tensor(best_val_loss)):.2f}")
    
    # Save training history
    history = {
        'config': {
            'num_layers': NUM_LAYERS,
            'nhead': NHEAD,
            'd_model': D_MODEL,
            'embedding': 'random',
            'epochs': NUM_EPOCHS
        },
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss,
        'total_time': total_time
    }
    
    history_path = os.path.join(os.path.dirname(__file__), 'output', 'json', 'training_history.json')
    os.makedirs(os.path.dirname(history_path), exist_ok=True)
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print("\n Model saved as: question5/best_model_4l_2h.pt")
    print(" Training history saved as: question5/output/json/training_history.json")


if __name__ == "__main__":
    train_best_config()
