import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DataParallel
from datasets import load_dataset
from collections import Counter
import spacy
import sacrebleu
import random
import time
import math
import numpy as np
from tqdm import tqdm
import pickle
import os
import argparse

# GPU Optimizations
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.cuda.empty_cache()

class CustomDataset(Dataset):
    def __init__(self, data, en_tokenizer, fr_tokenizer, en_vocab, fr_vocab, max_len=50):
        self.data = data
        self.en_tokenizer = en_tokenizer
        self.fr_tokenizer = fr_tokenizer
        self.en_vocab = en_vocab
        self.fr_vocab = fr_vocab
        self.max_len = max_len
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        en_text = item['translation']['en']
        fr_text = item['translation']['fr']
        
        # Tokenize sentences
        en_tokens = [tok.text.lower() for tok in self.en_tokenizer(en_text)]
        fr_tokens = [tok.text.lower() for tok in self.fr_tokenizer(fr_text)]
        
        # Reverse English tokens (source sentence) as per paper
        en_tokens = en_tokens[::-1]
        
        # Truncate if necessary
        en_tokens = en_tokens[:self.max_len-2]
        fr_tokens = fr_tokens[:self.max_len-2]
        
        # Add SOS and EOS tokens
        en_indices = [self.en_vocab['<sos>']] + [self.en_vocab.get(token, self.en_vocab['<unk>']) for token in en_tokens] + [self.en_vocab['<eos>']]
        fr_indices = [self.fr_vocab['<sos>']] + [self.fr_vocab.get(token, self.fr_vocab['<unk>']) for token in fr_tokens] + [self.fr_vocab['<eos>']]
        
        # Pad sequences
        en_indices = en_indices + [self.en_vocab['<pad>']] * (self.max_len - len(en_indices))
        fr_indices = fr_indices + [self.fr_vocab['<pad>']] * (self.max_len - len(fr_indices))
        
        return {
            'src': torch.tensor(en_indices),
            'trg': torch.tensor(fr_indices)
        }

def build_vocab(texts, tokenizer, max_vocab_size):
    """Build vocabulary with fixed size"""
    print(f"Building vocabulary with size {max_vocab_size}...")
    
    counter = Counter()
    for text in tqdm(texts, desc="Counting tokens"):
        tokens = [tok.text.lower() for tok in tokenizer(text)]
        counter.update(tokens)
    
    most_common = counter.most_common(max_vocab_size - 4)  # -4 for special tokens
    
    vocab = {'<pad>': 0, '<unk>': 1, '<sos>': 2, '<eos>': 3}
    for idx, (word, _) in enumerate(most_common, start=4):
        vocab[word] = idx
    
    print(f"Vocabulary size: {len(vocab)}")
    return vocab

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hidden_dim, n_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.rnn(embedded)
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hidden_dim, n_layers, dropout=dropout)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, cell):
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        prediction = self.fc_out(output.squeeze(0))
        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        hidden, cell = self.encoder(src)
        input = trg[0,:]
        
        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[t] if teacher_force else top1
        
        return outputs

def train_model(max_samples=None, force_rebuild_vocab=False, save_dir='checkpoints'):
    # Create directories
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs('cache', exist_ok=True)
    
    # Model hyperparameters
    BATCH_SIZE = 16
    ENC_EMB_DIM = 768
    DEC_EMB_DIM = 768
    HID_DIM = 768
    N_LAYERS = 4
    ENC_DROPOUT = 0.3
    DEC_DROPOUT = 0.3
    N_EPOCHS = 15
    CLIP = 5
    MAX_LEN = 50
    
    # Vocabulary sizes
    EN_VOCAB_SIZE = 160000
    FR_VOCAB_SIZE = 80000
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_gpu = torch.cuda.device_count()
    print(f"Using {n_gpu} GPUs" if n_gpu > 1 else f"Using {device}")
    
    # Load tokenizers
    print("Loading spaCy models...")
    en_nlp = spacy.load('en_core_web_sm')
    fr_nlp = spacy.load('fr_core_news_sm')
    
    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset("wmt14", 'fr-en')
    
    if max_samples is not None:
        dataset['train'] = dataset['train'].select(range(min(max_samples, len(dataset['train']))))
        dataset['validation'] = dataset['validation'].select(range(min(max_samples//10, len(dataset['validation']))))
    
    # Extract texts
    print("Extracting texts...")
    en_texts = [item['translation']['en'] for item in dataset['train']]
    fr_texts = [item['translation']['fr'] for item in dataset['train']]
    
    # Build vocabularies
    print("Building vocabularies...")
    en_vocab = build_vocab(en_texts, en_nlp, EN_VOCAB_SIZE)
    fr_vocab = build_vocab(fr_texts, fr_nlp, FR_VOCAB_SIZE)
    
    # Save vocabularies
    print("Saving vocabularies...")
    with open('cache/en_vocab.pkl', 'wb') as f:
        pickle.dump(en_vocab, f)
    with open('cache/fr_vocab.pkl', 'wb') as f:
        pickle.dump(fr_vocab, f)
    
    # Create datasets
    train_dataset = CustomDataset(dataset['train'], en_nlp, fr_nlp, en_vocab, fr_vocab, MAX_LEN)
    valid_dataset = CustomDataset(dataset['validation'], en_nlp, fr_nlp, en_vocab, fr_vocab, MAX_LEN)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE)
    
    # Initialize model
    enc = Encoder(EN_VOCAB_SIZE, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
    dec = Decoder(FR_VOCAB_SIZE, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)
    model = Seq2Seq(enc, dec, device).to(device)
    
    if n_gpu > 1:
        model = DataParallel(model)
    
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss(ignore_index=en_vocab['<pad>'])
    
    # Training loop
    best_valid_loss = float('inf')
    history = {
        'train_loss': [],
        'valid_loss': [],
        'train_ppl': [],
        'valid_ppl': []
    }
    
    for epoch in range(N_EPOCHS):
        start_time = time.time()
        model.train()
        epoch_loss = 0
        
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{N_EPOCHS}'):
            src = batch['src'].transpose(0, 1).to(device)
            trg = batch['trg'].transpose(0, 1).to(device)
            
            optimizer.zero_grad()
            output = model(src, trg)
            
            output_dim = output.shape[-1]
            output = output[1:].reshape(-1, output_dim)
            trg = trg[1:].reshape(-1)
            
            loss = criterion(output, trg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # Validation
        model.eval()
        valid_loss = 0
        
        with torch.no_grad():
            for batch in valid_loader:
                src = batch['src'].transpose(0, 1).to(device)
                trg = batch['trg'].transpose(0, 1).to(device)
                
                output = model(src, trg, 0)
                output_dim = output.shape[-1]
                output = output[1:].reshape(-1, output_dim)
                trg = trg[1:].reshape(-1)
                
                loss = criterion(output, trg)
                valid_loss += loss.item()
        
        # Calculate average losses and perplexity
        epoch_loss /= len(train_loader)
        valid_loss /= len(valid_loader)
        train_ppl = math.exp(epoch_loss)
        valid_ppl = math.exp(valid_loss)
        
        # Store history
        history['train_loss'].append(epoch_loss)
        history['valid_loss'].append(valid_loss)
        history['train_ppl'].append(train_ppl)
        history['valid_ppl'].append(valid_ppl)
        
        end_time = time.time()
        epoch_mins, epoch_secs = divmod(end_time - start_time, 60)
        
        # Save best model
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            model_path = os.path.join(save_dir, 'seq2seq-model-best.pt')
            
            # Save model with all necessary parameters
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': valid_loss,
                'en_vocab_size': EN_VOCAB_SIZE,
                'fr_vocab_size': FR_VOCAB_SIZE,
                'enc_emb_dim': ENC_EMB_DIM,
                'dec_emb_dim': DEC_EMB_DIM,
                'hid_dim': HID_DIM,
                'n_layers': N_LAYERS,
                'enc_dropout': ENC_DROPOUT,
                'dec_dropout': DEC_DROPOUT
            }, model_path)
            
            print(f"New best model saved to {model_path}")
        
        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {epoch_loss:.3f} | Train PPL: {train_ppl:7.3f}')
        print(f'\tValid Loss: {valid_loss:.3f} | Valid PPL: {valid_ppl:7.3f}')
    
    # Save final model and training history
    print("Saving final model and training history...")
    final_model_path = os.path.join(save_dir, 'seq2seq-model-final.pt')
    torch.save({
        'epoch': N_EPOCHS,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': valid_loss,
        'en_vocab_size': EN_VOCAB_SIZE,
        'fr_vocab_size': FR_VOCAB_SIZE,
        'enc_emb_dim': ENC_EMB_DIM,
        'dec_emb_dim': DEC_EMB_DIM,
        'hid_dim': HID_DIM,
        'n_layers': N_LAYERS,
        'enc_dropout': ENC_DROPOUT,
        'dec_dropout': DEC_DROPOUT
    }, final_model_path)
    
    history_path = os.path.join(save_dir, 'training_history.pkl')
    with open(history_path, 'wb') as f:
        pickle.dump(history, f)
    
    print(f"Training completed!")
    print(f"Final model saved to: {final_model_path}")
    print(f"Training history saved to: {history_path}")
    
    return model, history

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train translation model')
    parser.add_argument('--max_samples', type=int, default=None,
                      help='Maximum number of training samples to use (for testing)')
    parser.add_argument('--force_rebuild_vocab', action='store_true',
                      help='Force rebuilding vocabulary even if cache exists')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                      help='Directory to save model checkpoints')
    
    args = parser.parse_args()
    
    try:
        print("Starting training...")
        model, history = train_model(
            max_samples=args.max_samples,
            force_rebuild_vocab=args.force_rebuild_vocab,
            save_dir=args.save_dir
        )
        print("Training completed successfully!")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        raise