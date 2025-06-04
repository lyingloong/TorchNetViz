# TorchNetViz

**TorchNetViz** is a PyTorch model visualization toolkit designed to help developers and researchers better understand the structure and flow of their neural network models.

> 📚 **Documentation**: [https://lyingloong.github.io/Torchnetviz](https://lyingloong.github.io/Torchnetviz)  
> ⚠️ This project is currently under active development. Contributions and feedback are welcome!

## 📌 Features
- Interactive Model Graph Visualization
    - Visualize complex PyTorch model architectures with detailed layer information
    - Support for nested containers (e.g., Sequential), skip connections, and multi-input/output models
    - Export visualizations as HTML files or display them directly in Jupyter Notebooks
- Automatic Input Shape Inference
  - Dynamically infer input tensor shapes using randomized tensor testing
  - Enhanced with error recovery and intelligent shape guessing based on exception feedback
- Support for Diverse Model Architectures
  - Built-in support for common layers: Linear, CNN, RNN, LSTM, Transformer MLP blocks
  - Fully compatible with custom user-defined modules and architectures
- Modular & Extensible Design
  - Command-line interface for quick use
  - Easily integrable into existing PyTorch projects or workflows
- Advanced Graph Rendering Engine
  - Powered by Plotly for rich, interactive graph visualization
  - Multiple layout options:
    - Topological sorting-based layered layout
    - Nested force-directed layout (spring layout) for hierarchical structures

## 🧰 Requirements
- See [`requirements.txt`](requirements.txt) for dependencies

## 📁 Project Structure
```bash
.
├── models/                   # Reusable PyTorch model components and architectures
│   ├── DoubleCNN.py
│   ├── LSTM.py
│   ├── RNN.py
│   ├── TimeXer.py
│   └── custom_layers.py      # Custom reusable modules like LinearBlock, ResidualBlock, etc.
├── parser/                   # Model parsing and analysis utilities
│   ├── input_generator.py    # Generates dummy inputs based on inferred or specified shapes
│   ├── model_parser.py       # Main entry point for model structure parsing
│   ├── module_inspector.py   # Extracts layer/module details (type, parameters, source)
│   ├── shape_inference.py    # Automatic input shape inference via trial & error
│   ├── graph_extractor.py    # Analyzes traced model to extract connection graph
│   └── utils.py              # Helper functions for instantiation and TorchScript export
├── visualizer/               # Interactive visualization modules
│   └── graph_visualizer.py   # Plotly-based graph rendering with layout optimization
├── main.py                   # CLI entry point for model visualization
├── develop_logs/             # Development logs
├── output/                   # Output directory for generated JSON structures and HTML visuals
├── README.md                 # Project documentation
└── LICENSE                   # MIT License file
```


## 🚀 Getting Started

### Installation
```bash
pip install -r requirements.txt
```

### Usage
To visualize built-in models:
```bash
./scripts/powershell/Custom.ps1
```

## 📅 Development Log
See [Development Log](develop_logs/) for detailed daily updates and task tracking.

## 🤝 Contributing
Contributions are welcome! Please follow the standard fork → commit → pull request workflow.

## 📄 License
MIT License — see [`LICENSE`](LICENSE) for details.
