import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2

from models.video_encoder import VideoEncoder

class GradCAM3D:
    def __init__(self, model, target_layer_name):
        self.model = model
        self.target_layer = dict([*model.cnn.named_children()])[target_layer_name]
        self.gradients = None
        self.activations = None

        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate(self, input_tensor):
        self.model.zero_grad()
        output = self.model(input_tensor)
        score = output.sum()
        score.backward()

        pooled_grads = torch.mean(self.gradients, dim=[0, 2, 3, 4])
        activations = self.activations.squeeze(0)

        for i in range(activations.shape[0]):
            activations[i, :, :, :] *= pooled_grads[i]

        heatmap = torch.mean(activations, dim=0).cpu().numpy()
        heatmap = np.maximum(heatmap, 0)

        # Middle temporal slice
        mid_frame = heatmap[heatmap.shape[0] // 2]
        heatmap = cv2.resize(mid_frame, (64, 64))
        heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-8)

        return heatmap

def overlay_heatmap_on_frame(heatmap, frame):
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(frame, 0.5, heatmap_color, 0.5, 0)
    return overlay

if __name__ == "__main__": 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    video_enc = VideoEncoder().to(device)
    video_enc.eval()

    # Example video index - chang e as needed
    idx = 0

    # Load frames
    frames = np.load(f"data/features/frames_{idx}.npy")
    frames = frames.transpose(0, 3, 1, 2)
    torch_frames = torch.tensor(frames, dtype=torch.float32).unsqueeze(0).to(device)
    torch_frames = torch_frames.permute(0, 2, 1, 3, 4)

    # Middle frame in original data (to overlay)
    mid_index = frames.shape[0] // 2
    original_frame = frames[mid_index].transpose(1, 2, 0)  # (H, W, C)
    original_frame = cv2.resize(original_frame, (64, 64))
    original_frame = (original_frame * 255).astype(np.uint8)

    grad_cam = GradCAM3D(model=video_enc, target_layer_name="4")

    heatmap = grad_cam.generate(torch_frames)

    overlay = overlay_heatmap_on_frame(heatmap, original_frame)

    plt.figure(figsize=(8, 8))
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.title("Grad-CAM Overlay on Middle Video Frame")
    plt.axis("off")
    plt.show()
