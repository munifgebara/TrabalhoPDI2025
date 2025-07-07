from utils_pdi import (
    carregar_imagem,
    converter_para_hsv,
    normalizar_v,
    salvar_imagem,
    mostrar_imagem,
    mostrar_imagens_lado_a_lado, normalizar_v_com_limite, normalizar_v_com_limite_limitado,
    normalizar_v_com_limite_limitado_otsu
)
import cv2

import os

input_dir = "input"
output_dir = "output"

for filename in os.listdir(input_dir):
    print (filename)
    # Processa apenas arquivos com extensÃ£o .jpg (ignorando outros, se existirem)
    if filename.lower().endswith(".jpg"):
        caminho_entrada = os.path.join(input_dir, filename)
        base_name, ext = os.path.splitext(filename)  # separa nome e extensÃ£o
        caminho_saida = os.path.join(output_dir, base_name + "_hsv_gama_otsu" + ext)
        img_bgr = carregar_imagem(caminho_entrada)
        img_hsv = converter_para_hsv(img_bgr)
        img_hsv_norm = normalizar_v_com_limite_limitado_otsu(img_hsv)
        img_bgr_norm = cv2.cvtColor(img_hsv_norm, cv2.COLOR_HSV2BGR)
        salvar_imagem(img_bgr_norm, caminho_saida)
        mostrar_imagens_lado_a_lado(
            img_bgr, img_bgr_norm,
            f"Imagem Original - {filename}",
            f"V Normalizado - {filename}"
        )
