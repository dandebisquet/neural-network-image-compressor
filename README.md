# Neural Network Image Compressor (Coordinate MLP + Fourier Features)

This project is an experiment in **neural image compression** using a small neural network to represent an image as a function.

Instead of storing pixels directly, the program:
1. Converts each pixel location into **(x, y) coordinates**
2. Applies **Fourier feature encoding** to those coordinates
3. Trains an **MLP (multi-layer perceptron)** to predict the RGB value for every coordinate
4. Saves the trained model weights and optionally **compresses** them into a ZIP file

It includes a simple **Tkinter UI** to:
- select an image
- train the model to reconstruct it
- export/compress the trained model

## How it works (high level)
- **Input:** pixel coordinates in the range \[0,1\] × \[0,1\]
- **Encoding:** Fourier features (sin/cos at multiple frequencies) to help the network represent high-frequency detail
- **Model:** small MLP that maps encoded coordinates → RGB
- **Training objective:** MSE loss between predicted RGB and original RGB
- **“Compression”:** save the model weights (`.pth`) and zip them

## Demo / Features
- Tkinter GUI for interactive use
- Trains on a resized image (default 256×256)
- Reconstructs the image after training
- Exports a compressed model file (`compressed_model.zip`)

## Requirements
- Python 3.9+ recommended
- PyTorch, NumPy, Pillow (and optionally Matplotlib)

Install dependencies:
```bash
pip install -r requirements.txt
