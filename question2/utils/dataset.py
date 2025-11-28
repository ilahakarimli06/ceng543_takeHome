# question2/utils/dataset.py
import torch
from torch.utils.data import Dataset
PAD_ID = 0  

def read_ids(path):
    with open(path, "r", encoding="utf-8") as f:
        return [[int(t) for t in line.strip().split()] for line in f if line.strip()]  # Read token ID lists from file

class ParallelIdsDataset(Dataset):
    def __init__(self, source_path, target_path):
        self.source = read_ids(source_path)  # Read source sentences
        self.target = read_ids(target_path)  # Read target sentences
        self.pairs = [(source, target) for source, target in zip(self.source, self.target) 
                  if len(source) > 0 and len(target) > 1]  # Skip empty or too short sentences

    def __getitem__(self, idx):
        source, target = self.pairs[idx]
        target_input = target[:-1]          # target input without <eos>
        target_output = target[1:]          # target output without <sos>
        return (torch.tensor(source), torch.tensor(target_input), torch.tensor(target_output))
    
    def __len__(self):
        return len(self.pairs)

def collate_pad(batch, pad_id=PAD_ID):
    sources, target_inputs, target_outputs = zip(*batch)
    source = torch.nn.utils.rnn.pad_sequence(sources, batch_first=True, padding_value=pad_id)  # Pad sources
    target_input = torch.nn.utils.rnn.pad_sequence(target_inputs, batch_first=True, padding_value=pad_id)  # Pad target inputs
    target_output = torch.nn.utils.rnn.pad_sequence(target_outputs, batch_first=True, padding_value=pad_id)  # Pad target outputs
    source_mask = (source == pad_id)   # Create mask for source
    target_mask = (target_input == pad_id)  # Create mask for target
    return {"src": source, "target_input": target_input, "tgt_out": target_output,
            "source_mask": source_mask, "target_mask": target_mask}  # Return padded batch and masks