# 🤖 Neural Machine Translation with Sequence-to-Sequence Learning

## 🎯 Project Overview
A PyTorch implementation of Neural Machine Translation (NMT) using Sequence-to-Sequence architecture with LSTM networks. This project implements the approach described in the paper "Sequence to Sequence Learning with Neural Networks" (Sutskever et al., 2014), with optimizations for modern GPU acceleration.

## ✨ Key Features
- 🏗️ **Architecture**: Multi-layer LSTM-based Encoder-Decoder model
- 📈 **Performance**: Achieved BLEU score of 23 on English-French translation task
- 🚀 **GPU Optimization**: CUDA-accelerated with support for RTX series GPUs
- 🔄 **Data Processing**: Efficient handling of WMT'14 dataset with dynamic batching
- 📚 **Vocabulary Management**: Implemented frequency-based vocabulary construction

## 🛠️ Technical Implementation
### 🧬 Model Architecture
```python
# Core model parameters
BATCH_SIZE = 16
ENC_EMB_DIM = 768
DEC_EMB_DIM = 768
HID_DIM = 768
N_LAYERS = 4
N_EPOCHS = 15
CLIP = 5
```

### 🔋 Key Components
- 🔍 **Encoder**: Multi-layer LSTM with dropout for regularization
- 🎯 **Decoder**: Multi-layer LSTM with output projection layer
- 📊 **Embedding**: 768-dimensional word embeddings
- ⚡ **Optimization**: Adam optimizer with gradient clipping
- 📉 **Loss Function**: Cross-entropy with padding mask

### 💻 GPU Optimizations
```python
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```

## 📊 Results
- 📈 **BLEU Score**: 23.0 on WMT'14 test set
- ⚡ **Training Time**: Optimized for RTX 3060 GPU
- 💾 **Memory Efficiency**: Batch size optimization for 6GB VRAM

## 🚀 Installation and Setup
1. Clone the repository:
```bash
git clone https://github.com/yourusername/nmt-project.git
cd nmt-project
```

2. Install requirements:
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m spacy download fr_core_news_sm
```

3. Prepare the environment:
```bash
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

## 🎮 Usage
### 🏃‍♂️ Training
```bash
python train.py --max_samples 1000 --force_rebuild_vocab
```

### 🔄 Translation
```bash
python utils.py --model_path checkpoints/seq2seq-model-final.pt --mode translate --text "Hello, how are you?"
```

### 📊 Evaluation
```bash
python utils.py --model_path checkpoints/seq2seq-model-final.pt --mode evaluate --num_samples 100
```

## 📁 Project Structure
```
nmt-project/
│
├── train.py           # Training script
├── utils.py           # Utility functions and inference
├── requirements.txt   # Project dependencies
├── checkpoints/       # Saved models
├── cache/            # Vocabulary cache
└── README.md         # Project documentation
```

## 🔬 Technical Details
### 🔄 Data Processing Pipeline
1. 📝 **Tokenization**: SpaCy-based tokenization for both languages
2. 📚 **Vocabulary Building**: Frequency-based with special tokens
3. 🔄 **Batch Processing**: Dynamic batching with padding
4. 🔀 **Source Sentence Reversal**: Implemented as per original paper

### 🎯 Model Features
- 📊 **Gradient Clipping**: Prevents exploding gradients
- 🎓 **Teacher Forcing**: Implemented during training
- 🎲 **Dropout Regularization**: Prevents overfitting
- 💾 **Checkpoint Management**: Saves best and final models

## 🔮 Future Improvements
- ⚡ Implementation of attention mechanism
- 🔍 Beam search for better inference
- 📈 Learning rate scheduling
- ⚖️ Layer normalization
- 🔄 Bidirectional encoder

## 📋 Requirements
- 🐍 Python 3.8+
- 🔥 PyTorch 2.0+
- 💻 CUDA compatible GPU
- 💾 6GB+ VRAM
- 🔤 spaCy
- 📊 sacrebleu
- 📈 tqdm

## 📚 Citation
```bibtex
@article{sutskever2014sequence,
  title={Sequence to Sequence Learning with Neural Networks},
  author={Sutskever, Ilya and Vinyals, Oriol and Le, Quoc V},
  journal={arXiv preprint arXiv:1409.3215},
  year={2014}
}
```

## 📄 License
MIT License

## 🙏 Acknowledgments
- 📊 WMT'14 dataset providers
- 🔥 PyTorch team for the deep learning framework
- 📚 Original Seq2Seq paper authors


## 🤝 Contributing
Feel free to:
- 🐛 Report bugs
- 💡 Suggest features
- 🔀 Submit PRs
