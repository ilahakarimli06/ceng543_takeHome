import torch
import torch.nn as nn
from transformers import AutoModel

#Usage is looked from PyTorch Documentation
class BiLSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_classes):
        
        #Initialize a bidirectional LSTM for sequence classification.
        
        super().__init__()
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        # Forward pass for BiLSTM. Expects dict with 'inputs' and 'lengths'.
        # Returns logits for each class.
        
        # x is dict: {"inputs": [Batch, Length, Dim], "lengths": [Batch]}
        embeddings = x["inputs"]
        lengths = x["lengths"]
        
        out, _ = self.lstm(embeddings)  # [Batch, Length, hidden*2]
        
        # Get the actual last timestep
        batch_size = out.size(0)
        last_hidden = out[torch.arange(batch_size), lengths - 1]
        
        return self.fc(last_hidden)
    
    def get_features(self, x):
        # Extract latent representations (hidden states) for visualization.
        # Returns the last hidden state for each sequence.
        embeddings = x["inputs"]
        lengths = x["lengths"]
        
        out, _ = self.lstm(embeddings)
        batch_size = out.size(0)
        return out[torch.arange(batch_size), lengths - 1]


class BiGRU(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_classes):
        # Initialize a bidirectional GRU for sequence classification.
        super().__init__()
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        # x is dict: {"inputs": [Batch, Length, Dim], "lengths": [Batch]}
        embeddings = x["inputs"]
        lengths = x["lengths"]
        
        out, _ = self.gru(embeddings)  # [Batch, Length, hidden*2]
        
        # Get the actual last timestep
        batch_size = out.size(0)
        last_hidden = out[torch.arange(batch_size), lengths - 1]
        
        return self.fc(last_hidden)
    
    def get_features(self, x):
        # Extract latent representations (hidden states) for visualization.
        embeddings = x["inputs"]
        lengths = x["lengths"]
        
        out, _ = self.gru(embeddings)
        batch_size = out.size(0)
        return out[torch.arange(batch_size), lengths - 1]


# Contextual (BERT) + BiLSTM
class BERTBiLSTM(nn.Module):
    def __init__(self, bert_model_name, hidden_dim, num_classes):
        super().__init__()
        self.bert = AutoModel.from_pretrained(bert_model_name)
        bert_hidden_size = self.bert.config.hidden_size
        self.lstm = nn.LSTM(bert_hidden_size, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
    
    def forward(self, x):
        # x is a dict: {"input_ids", "attention_mask", ...}
        bert_output = self.bert(**x)
        embeddings = bert_output.last_hidden_state  # [batch, seq_len, bert_hidden]
        out, _ = self.lstm(embeddings)
        
    
        attention_mask = x["attention_mask"]
        lengths = attention_mask.sum(dim=1)  # [batch]
        batch_size = out.size(0)
        last_hidden = out[torch.arange(batch_size), lengths - 1]
        
        return self.fc(last_hidden)
    
    def get_features(self, x):
        # Extract latent representations (hidden states) for visualization.
        # Returns the last hidden state for each sequence.
        bert_output = self.bert(**x)
        embeddings = bert_output.last_hidden_state
        out, _ = self.lstm(embeddings)
        
        attention_mask = x["attention_mask"]
        lengths = attention_mask.sum(dim=1)
        batch_size = out.size(0)
        return out[torch.arange(batch_size), lengths - 1]


# Contextual (BERT) + BiGRU
class BERTBiGRU(nn.Module):
    def __init__(self, bert_model_name, hidden_dim, num_classes):
        super().__init__()
        self.bert = AutoModel.from_pretrained(bert_model_name)
        bert_hidden_size = self.bert.config.hidden_size
        self.gru = nn.GRU(bert_hidden_size, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
    
    def forward(self, x):
        bert_output = self.bert(**x)
        embeddings = bert_output.last_hidden_state 
        out, _ = self.gru(embeddings)
        

        attention_mask = x["attention_mask"]
        lengths = attention_mask.sum(dim=1)
        batch_size = out.size(0)
        last_hidden = out[torch.arange(batch_size), lengths - 1]
        
        return self.fc(last_hidden)
    
    def get_features(self, x):
        """Extract latent representations (hidden states) for visualization."""
        bert_output = self.bert(**x)
        embeddings = bert_output.last_hidden_state
        out, _ = self.gru(embeddings)
        
        attention_mask = x["attention_mask"]
        lengths = attention_mask.sum(dim=1)
        batch_size = out.size(0)
        return out[torch.arange(batch_size), lengths - 1]
