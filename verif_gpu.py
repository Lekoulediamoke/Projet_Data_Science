import torch

print("CUDA disponible :", torch.cuda.is_available())
print("Périphérique GPU :", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Aucun GPU détecté")
