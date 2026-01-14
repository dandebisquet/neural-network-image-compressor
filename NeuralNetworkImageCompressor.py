import os
import zipfile
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import threading
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- Fourier Feature Encoding ----------
def fourier_encode(x, num_bands=6, max_freq=8.0):
    x = x * 2 * torch.pi
    freqs = 2.0 ** torch.arange(num_bands, dtype=torch.float32, device=x.device) * max_freq
    freqs = freqs[None, None, :]
    x_proj = x[..., None] * freqs
    x_proj = x_proj.view(x.shape[0], -1)
    return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

# ---------- Neural Network ----------
class ImageMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 3),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# ---------- Core Logic ----------
def load_image(image_path, size=(256, 256)):
    img = Image.open(image_path).convert("RGB")
    img = img.resize(size, Image.LANCZOS)
    img_np = np.asarray(img) / 255.0

    H, W = size
    xs = np.linspace(0, 1, W)
    ys = np.linspace(0, 1, H)
    x_coords, y_coords = np.meshgrid(xs, ys)
    coords = np.stack([x_coords, y_coords], axis=-1).reshape(-1, 2)
    colors = img_np.reshape(-1, 3)

    coords_tensor = torch.tensor(coords, dtype=torch.float32)
    colors_tensor = torch.tensor(colors, dtype=torch.float32)
    return coords_tensor, colors_tensor, img_np, (H, W)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compress_model(model, zip_path=None, temp_model_path="model_temp.pth"):
    import os
    if zip_path is None:
        os.makedirs("compressed", exist_ok=True)
        zip_path = os.path.join("compressed", "compressed_model.zip")
        
    print("[INFO] Starting model compression...")
    try:
        model = model.half().to("cpu")
        torch.save(model.state_dict(), temp_model_path)
        print(f"[INFO] Saved model to temporary path: {temp_model_path}")
    except Exception as e:
        print(f"[ERROR] Failed to save model: {e}")
        return
    
    try:
        with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(temp_model_path, arcname=os.path.basename(temp_model_path))
        print(f"[INFO] Compressed model to: {zip_path}")
    except Exception as e:
        print(f"[ERROR] Failed to compress model: {e}")
        return
    
    try:
        os.remove(temp_model_path)
        print(f"[INFO] Deleted temporary file: {temp_model_path}")
    except Exception as e:
        print(f"[WARNING] Failed to delete temporary file: {e}")

    print(f"[INFO] Compression complete! Find your zip at: {os.path.abspath(zip_path)}")



# ---------- UI Application ----------
class ImageReconstructionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Reconstructor")

        self.img_label = tk.Label(root)
        self.img_label.pack()

        self.status = tk.Label(root, text="Status: Waiting", fg="blue")
        self.status.pack()

        tk.Button(root, text="Select Image", command=self.select_image).pack()
        tk.Button(root, text="Reconstruct", command=self.start_reconstruction).pack()
        tk.Button(root, text="Compress & Save Model", command=self.save_model).pack()

        self.original = None
        self.coords = None
        self.colors = None
        self.model = None
        self.H = None
        self.W = None

    def select_image(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg")])
        if not path:
            return
        self.coords, self.colors, self.original, (self.H, self.W) = load_image(path)
        self.display_image(self.original)
        self.status.config(text="Image loaded")

    def display_image(self, np_img):
        img = (np_img * 255).astype(np.uint8)
        img = Image.fromarray(img)
        img = ImageTk.PhotoImage(img)
        self.img_label.config(image=img)
        self.img_label.image = img

    def start_reconstruction(self):
        if self.coords is None:
            messagebox.showerror("Error", "Load an image first!")
            return
        threading.Thread(target=self.reconstruct).start()

    def reconstruct(self):
        self.status.config(text="Training...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        coords = self.coords.to(device)
        colors = self.colors.to(device)
        encoded_coords = fourier_encode(coords)
        input_dim = encoded_coords.shape[1]

        self.model = ImageMLP(input_dim).to(device)
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        for epoch in range(1000):
            optimizer.zero_grad()
            preds = self.model(encoded_coords)
            loss = criterion(preds, colors)
            loss.backward()
            optimizer.step()
            if epoch % 100 == 0:
                self.status.config(text=f"Epoch {epoch}, Loss: {loss.item():.6f}")

        with torch.no_grad():
            out = self.model(encoded_coords).cpu()
            out = torch.clamp(out, 0, 1)
            result = out.view(self.H, self.W, 3).numpy()
        self.display_image(result)
        self.status.config(text="Done!")

    def save_model(self):
        if self.model is None:
            messagebox.showerror("Error", "No model to save.")
            return
        compress_model(self.model, "compressed_model.zip")
        self.status.config(text="Model compressed and saved to 'compressed_model.zip'")

# ---------- Run App ----------
if __name__ == "__main__":
    root = tk.Tk()
    app = ImageReconstructionApp(root)
    root.mainloop()

