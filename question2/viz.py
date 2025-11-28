import random
import torch
import matplotlib.pyplot as plt
import sentencepiece as spm
from pathlib import Path

from models.seq2seq import Seq2Seq
from utils.dataset import PAD_ID, read_ids
import sys

@torch.no_grad()
def greedy_decode_with_attention(model, src_ids, sp, device, max_len=50):
    """Decode and return tokens + attention weights"""
    model.eval()
    bos_id, eos_id = sp.bos_id(), sp.eos_id()
    
    src = torch.tensor([src_ids], device=device, dtype=torch.long)
    source_mask = (src == PAD_ID)
    enc_outs, (h, c) = model.encoder(src, source_mask)
    
    tokens, attentions = [], []
    prev_token = torch.tensor([bos_id], device=device, dtype=torch.long)
    
    for _ in range(max_len):
        emb = model.decoder.drop(model.decoder.emb(prev_token)).unsqueeze(1)
        context, attn = model.decoder.attn(h[-1], enc_outs, source_mask)
        context = context.unsqueeze(1)
        
        rnn_in = torch.cat([emb, context], dim=-1)
        out, (h, c) = model.decoder.rnn(rnn_in, (h, c))
        
        logits = model.decoder.fc_out(torch.cat([out, context, emb], dim=-1))
        next_token = logits.argmax(-1).squeeze()
        
        tokens.append(next_token.item())
        attentions.append(attn.squeeze(0).cpu())
        
        if next_token.item() == eos_id:
            break
        prev_token = next_token.unsqueeze(0)
    
    attn_matrix = torch.stack(attentions, dim=0).numpy()  # (tgt_len, src_len)
    return tokens, attn_matrix


def plot_attention(attn, src_tokens, tgt_tokens, title, save_path):
    """Plot and save attention heatmap"""
    fig_width = max(8, len(src_tokens) * 0.4)
    fig_height = max(6, len(tgt_tokens) * 0.3)
    
    plt.figure(figsize=(fig_width, fig_height))
    plt.imshow(attn, cmap='Blues', aspect='auto', interpolation='nearest')
    
    plt.xticks(range(len(src_tokens)), src_tokens, rotation=90, fontsize=9)
    plt.yticks(range(len(tgt_tokens)), tgt_tokens, fontsize=9)
    plt.xlabel('Source Tokens', fontsize=11)
    plt.ylabel('Target Tokens', fontsize=11)
    plt.title(title, fontsize=12)
    plt.colorbar(label='Attention Weight')
    plt.tight_layout()
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()


def visualize_attention(model_path, output_dir, 
                        spm_model='question2/data/spm_shared_unigram.model',
                        test_src='question2/data/tokenized/test.en.ids',
                        test_tgt='question2/data/tokenized/test.de.ids',
                        num_samples=5):
    """Visualize attention maps for samples"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['cfg']
    model = Seq2Seq(
        src_vocab_size=config.get('src_vocab', 8000),
        tgt_vocab_size=config.get('tgt_vocab', 8000),
        emb_dim=config.get('embedding_dim', config.get('emb_dim', 256)),
        enc_hid=config.get('encoder_hidden', config.get('enc_hid', 256)),
        dec_hid=config.get('decoder_hidden', config.get('dec_hid', 256)),
        num_layers=config.get('num_layers', 2),
        attn_type=config.get('attention_type', config.get('attn_type', 'additive')),
        dropout=config.get('dropout', 0.1)
    ).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    attention_type = config.get('attention_type', config.get('attn_type', 'additive'))
    
    # Load tokenizer and data
    sp = spm.SentencePieceProcessor(model_file=spm_model)
    src_seqs = read_ids(test_src)
    tgt_seqs = read_ids(test_tgt)
    
    # Sample random examples (handle small datasets)
    random.seed(42)
    num_samples = min(num_samples, len(src_seqs))
    if num_samples == 0:
        print("Warning: No samples available for visualization")
        return
    indices = random.sample(range(len(src_seqs)), num_samples)
    
    for i, idx in enumerate(indices, 1):
        src_ids = src_seqs[idx]
        tgt_ids = tgt_seqs[idx]
        
        # Decode with attention
        pred_ids, attn_matrix = greedy_decode_with_attention(model, src_ids, sp, device)
        
        # Convert to tokens
        src_tokens = [sp.id_to_piece(int(id)) for id in src_ids]
        pred_tokens = [sp.id_to_piece(int(id)) for id in pred_ids]
        
        # Trim attention matrix
        attn_view = attn_matrix[:len(pred_tokens), :len(src_tokens)]
        
        # Create title
        source_text = sp.decode(src_ids)
        reference_text = sp.decode(tgt_ids)
        predicted_text = sp.decode(pred_ids)
        title = f'{attention_type.upper()} - Sample {i}\\nSource: {source_text}\\nReference: {reference_text}\\nPrediction: {predicted_text}'
        
        # Save plot
        save_path = Path(output_dir) / f'sample_{i:02d}_idx{idx}.png'
        plot_attention(attn_view, src_tokens, pred_tokens, title, save_path)
        
        print(f'[{i}/{num_samples}] Saved: {save_path}')


if __name__ == '__main__':
    
    if len(sys.argv) != 3:
        print('Usage: python viz.py <model_path> <output_dir>')
        print('Example: python question2/viz.py question2/output/additive_model.pt question2/output/attn_maps/additive')
        sys.exit(1)
    
    model_path = sys.argv[1]
    output_dir = sys.argv[2]
    
    visualize_attention(model_path, output_dir)
