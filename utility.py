
import torch 
import torch.nn as nn
import spacy
import pickle
from datasets import load_dataset
import sacrebleu
from tqdm import tqdm
import argparse

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
            top1 = output.argmax(1)
            input = top1
        
        return outputs

class Translator:
    def __init__(self, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model and vocab
        self.checkpoint = torch.load(model_path, map_location=self.device)
        with open('cache/en_vocab.pkl', 'rb') as f:
            self.en_vocab = pickle.load(f)
        with open('cache/fr_vocab.pkl', 'rb') as f:
            self.fr_vocab = pickle.load(f)
            
        self.en_vocab_reverse = {v: k for k, v in self.en_vocab.items()}
        self.fr_vocab_reverse = {v: k for k, v in self.fr_vocab.items()}
        
        self.en_nlp = spacy.load('en_core_web_sm')
        self.model = self._build_model()
        self.model.eval()
    
    def _build_model(self):
        encoder = Encoder(
            input_dim=self.checkpoint['en_vocab_size'],
            emb_dim=self.checkpoint['enc_emb_dim'],
            hidden_dim=self.checkpoint['hid_dim'],
            n_layers=self.checkpoint['n_layers'],
            dropout=self.checkpoint['enc_dropout']
        )
        
        decoder = Decoder(
            output_dim=self.checkpoint['fr_vocab_size'],
            emb_dim=self.checkpoint['dec_emb_dim'],
            hidden_dim=self.checkpoint['hid_dim'],
            n_layers=self.checkpoint['n_layers'],
            dropout=self.checkpoint['dec_dropout']
        )
        
        model = Seq2Seq(encoder, decoder, self.device)
        model.load_state_dict(self.checkpoint['model_state_dict'])
        return model.to(self.device)

    def translate(self, text):
        self.model.eval()
        
        # Tokenize and reverse
        tokens = [tok.text.lower() for tok in self.en_nlp(text)]
        tokens = tokens[::-1]
        
        # Convert to indices
        token_ids = [self.en_vocab['<sos>']]
        token_ids.extend([self.en_vocab.get(token, self.en_vocab['<unk>']) for token in tokens])
        token_ids.append(self.en_vocab['<eos>'])
        
        src_tensor = torch.LongTensor(token_ids).unsqueeze(1).to(self.device)
        
        with torch.no_grad():
            hidden, cell = self.model.encoder(src_tensor)
            trg_idx = [self.fr_vocab['<sos>']]
            
            for _ in range(50):
                trg_tensor = torch.LongTensor([trg_idx[-1]]).to(self.device)
                output, hidden, cell = self.model.decoder(trg_tensor, hidden, cell)
                pred_token = output.argmax(1).item()
                trg_idx.append(pred_token)
                
                if pred_token == self.fr_vocab['<eos>']:
                    break
        
        translated_tokens = [self.fr_vocab_reverse[i] for i in trg_idx]
        translated_tokens = [token for token in translated_tokens 
                           if token not in ['<sos>', '<eos>', '<pad>', '<unk>']]
        return ' '.join(translated_tokens)

    def get_bleu(self, num_samples=100):
        dataset = load_dataset("wmt14", "fr-en")['test']
        samples = list(dataset)[:num_samples]
        
        references = []
        hypotheses = []
        
        for sample in samples:
            try:
                src_text = sample['translation']['en']
                ref_text = sample['translation']['fr']
                translation = self.translate(src_text)
                
                references.append([ref_text])
                hypotheses.append(translation)
            except:
                continue
                
        bleu = sacrebleu.corpus_bleu(hypotheses, references)
        return bleu.score

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--mode', choices=['translate', 'bleu'], required=True)
    parser.add_argument('--text', help='Text to translate')
    parser.add_argument('--num_samples', type=int, default=100)
    args = parser.parse_args()
    
    translator = Translator(args.model_path)
    
    if args.mode == 'translate':
        if not args.text:
            print("Please provide text to translate with --text")
            return
        translation = translator.translate(args.text)
        print(f"Translation: {translation}")
        
    elif args.mode == 'bleu':
        bleu = translator.get_bleu(args.num_samples)
        print(f"BLEU Score: {bleu:.2f}")

if __name__ == "__main__":
    main()
