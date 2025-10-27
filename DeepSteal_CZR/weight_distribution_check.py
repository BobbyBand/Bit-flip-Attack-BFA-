import torch
import matplotlib.pyplot as plt

# Load float model weights
state_f = torch.load("victim_vgg16_cifar10_float.pt", map_location="cpu")
weights_f = state_f["features.18.weight"].flatten().float()

# Load int8 model weights
state_q = torch.load("victim_vgg16_cifar10_int8.pt", map_location="cpu")
weights_q = state_q["features.18.weight::qint8"].flatten().float()  # may be torch.int8

# If weights_q is int8, convert to float for plotting
if weights_q.dtype == torch.int8:
    weights_q = weights_q.to(torch.float32)

# Optional: dequantize if scale/zero_point are available
# For example, if you stored them during quantization:
# scale = state_q["features.0.scale"]
# zero_point = state_q["features.0.zero_point"]
# weights_q = scale * (weights_q - zero_point)

# Plot both histograms
plt.figure(figsize=(10,5))
plt.hist(weights_f.numpy(), bins=100, alpha=0.6, label="float32 weights")
plt.hist(weights_q.numpy(), bins=100, alpha=0.6, label="int8 weights")
plt.legend()
plt.title("Weight Distribution Comparison")
plt.xlabel("Weight value")
plt.xlim(-127, 128) 
plt.ylabel("Frequency")
plt.ylim(0, 20)
plt.savefig("weights_hist.png")
