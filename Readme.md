# TorchNetViz

**TorchNetViz** is a PyTorch model visualization toolkit designed to help developers and researchers better understand the structure and flow of their neural network models.

> ⚠️ This project is currently under active development. Contributions and feedback are welcome!

## 📌 Features
- Visualize PyTorch model graphs with layer details
- Interactive web-based UI for model exploration
- Support for common model architectures including:
  - Linear blocks
  - Convolutional layers
  - Residual blocks (ResNet-style)
  - Transformer MLP blocks
- Modular logging system for tracking development progress

## 🧰 Requirements
- See [`requirements.txt`](requirements.txt) for dependencies

## 📁 Project Structure
```bash
.
├── models/               # Custom model components
│   └── custom_layers.py  # Reusable PyTorch modules
├── parser/               # Model parsing utilities
├── output/               # Generated visualizations and logs
├── develop_logs/         # Development logs (not committed)
└── README.md             # This file
```


## 🚀 Getting Started

### Installation
```bash
pip install -r requirements.txt
```

### Usage
To visualize a model:
```bash
python main.py --model=path/to/your/model.py
```

## 📅 Development Log
See [Development Log](develop_logs/) for detailed daily updates and task tracking.

## 🤝 Contributing
Contributions are welcome! Please follow the standard fork → commit → pull request workflow.

## 📄 License
MIT License — see [`LICENSE`](LICENSE) for details.