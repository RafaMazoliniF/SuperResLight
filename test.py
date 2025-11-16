import torch
import torchvision.transforms.functional as F
from torchvision.io import read_image
from torchvision.utils import save_image

# Carrega a imagem (o resultado é uint8 [0, 255])
equalized_tensor = read_image("RELLISUR-Dataset/Test/LLLR/00018-4.5.png") 

equalized_tensor = F.adjust_gamma(equalized_tensor, 0.8)
equalized_tensor = F.adjust_brightness(equalized_tensor, 1.2)
equalized_tensor = F.adjust_contrast(equalized_tensor, 1.2)
# equalized_tensor = F.(equalized_tensor)

# --- Correção Aqui ---
# Converta o tensor uint8 [0, 255] para float [0, 1] antes de salvar
tensor_para_salvar = equalized_tensor.float() / 255.0
# --- Fim da Correção ---

# Salva o resultado (agora recebe o tipo float esperado)
save_image(tensor_para_salvar, "imagem_equalizada.jpg")