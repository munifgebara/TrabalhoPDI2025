import os
import numpy as np
import cv2
from matplotlib import pyplot as plt

def carregar_imagem(caminho):
    imagem = cv2.imread(caminho)
    if imagem is None:
        raise FileNotFoundError(f"Imagem não encontrada: {caminho}")
    return imagem

def converter_para_hsv(imagem_bgr):
    return cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)

def salvar_imagem(imagem_bgr, caminho_saida):
    os.makedirs(os.path.dirname(caminho_saida), exist_ok=True)
    cv2.imwrite(caminho_saida, imagem_bgr)

def mostrar_imagem(imagem_bgr, titulo="Imagem"):
    imagem_rgb = cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2RGB)
    plt.imshow(imagem_rgb)
    plt.title(titulo)
    plt.axis("off")
    plt.show()

def mostrar_imagens_lado_a_lado(img1_bgr, img2_bgr, titulo1="Original", titulo2="Processada"):
    img1_rgb = cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2RGB)
    img2_rgb = cv2.cvtColor(img2_bgr, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img1_rgb)
    plt.title(titulo1)
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(img2_rgb)
    plt.title(titulo2)
    plt.axis("off")
    plt.tight_layout()
    plt.show()

def normalizar_v(hsv_img):
    h = hsv_img[:, :, 0]
    s = hsv_img[:, :, 1]
    v = hsv_img[:, :, 2]
    v_min = v.min()
    v_max = v.max()
    print(f"v_min: {v_min:.2f}, v_max: {v_max:.2f}")
    if v_max - v_min == 0:
        v_norm = np.zeros_like(v)
    else:
        v_norm = (v - v_min) / (v_max - v_min) * 255
    hsv_normalizado = np.stack([h, s, v_norm], axis=2).astype(np.uint8)
    return hsv_normalizado


def normalizar_v_com_limite(hsv_img, limite_v):
    h = hsv_img[:, :, 0]
    s = hsv_img[:, :, 1]
    v = hsv_img[:, :, 2]
    v_min = v.min()
    v_max = v.max()
    print(f"v_min: {v_min:.2f}, v_max: {v_max:.2f}, limite: {limite_v}")
    v_norm = v.copy()
    mascara = v < limite_v
    if np.any(mascara):
        v_sub = v[mascara]
        v_sub_min = v_sub.min()
        v_sub_max = v_sub.max()
        if v_sub_max - v_sub_min == 0:
            v_norm[mascara] = 0
        else:
            v_norm[mascara] = ((v_sub - v_sub_min) / (v_sub_max - v_sub_min) * 255)
    hsv_normalizado = np.stack([h, s, v_norm], axis=2).astype(np.uint8)
    return hsv_normalizado

def normalizar_v_com_limite_limitado(hsv_img, limite_v):
    h = hsv_img[:, :, 0]
    s = hsv_img[:, :, 1]
    v = hsv_img[:, :, 2]
    v_min = v.min()
    v_max = v.max()
    print(f"v_min: {v_min:.2f}, v_max: {v_max:.2f}, limite: {limite_v}")
    v_norm = v.copy()
    mascara = v < limite_v
    if np.any(mascara):
        v_sub = v[mascara]
        v_sub_min = v_sub.min()
        v_sub_max = v_sub.max()
        if v_sub_max - v_sub_min == 0:
            v_norm[mascara] = 0
        else:
            v_norm[mascara] = ((v_sub - v_sub_min) / (v_sub_max - v_sub_min) * limite_v * 1.3)
    hsv_normalizado = np.stack([h, s, v_norm], axis=2).astype(np.uint8)
    return hsv_normalizado

def limiar_otsu(v):
    v = v.astype(np.uint8).flatten()
    hist, bins = np.histogram(v, bins=256, range=(0, 256))
    total = v.size
    soma_total = np.dot(np.arange(256), hist)  # soma de i * hist[i]
    soma_fundo = 0
    peso_fundo = 0
    max_var_entre_classes = 0
    limiar_otimo = 0

    for t in range(256):
        peso_fundo += hist[t]
        if peso_fundo == 0:
            continue
        peso_frente = total - peso_fundo
        if peso_frente == 0:
            break

        soma_fundo += t * hist[t]
        media_fundo = soma_fundo / peso_fundo
        media_frente = (soma_total - soma_fundo) / peso_frente

        var_entre_classes = peso_fundo * peso_frente * (media_fundo - media_frente) ** 2

        if var_entre_classes > max_var_entre_classes:
            max_var_entre_classes = var_entre_classes
            limiar_otimo = t

    return limiar_otimo


def normalizar_v_com_limite_limitado_otsu(hsv_img):
    h = hsv_img[:, :, 0]
    s = hsv_img[:, :, 1]
    v = hsv_img[:, :, 2]

    limite_v = limiar_otsu(v)

    print ("limite_otsu: ", limite_v)

    v_min = v.min()
    v_max = v.max()
    print(f"v_min: {v_min:.2f}, v_max: {v_max:.2f}, limite: {limite_v}")
    v_norm = v.copy()
    mascara = v < limite_v
    if np.any(mascara):
        v_sub = v[mascara]
        v_sub_min = v_sub.min()
        v_sub_max = v_sub.max()
        if v_sub_max - v_sub_min == 0:
            v_norm[mascara] = 0
        else:
            v_norm[mascara] = ((v_sub - v_sub_min) / (v_sub_max - v_sub_min) * limite_v * 1.8)
    hsv_normalizado = np.stack([h, s, v_norm], axis=2).astype(np.uint8)
    return hsv_normalizado


def corrigir_gama_duplo(hsv_img, gama_baixo, gama_alto):
    h = hsv_img[:, :, 0]
    s = hsv_img[:, :, 1]
    v = hsv_img[:, :, 2].astype(np.float32)
    limiar = limiar_otsu(v)
    print(f"Limiar de Otsu: {limiar} , gama alto: {gama_alto} gama baxo: {gama_baixo}")
    v_norm = v / 255.0
    mascara_baixo = v < limiar
    mascara_alto = ~mascara_baixo
    v_corrigido = np.zeros_like(v_norm)
    v_corrigido[mascara_baixo] = np.power(v_norm[mascara_baixo],  gama_baixo)
    v_corrigido[mascara_alto] = np.power(v_norm[mascara_alto], gama_alto)
    v_corrigido = np.clip(v_corrigido * 255.0, 0, 255)
    hsv_corrigido = np.stack([h, s, v_corrigido], axis=2).astype(np.uint8)
    return hsv_corrigido

def hsv_para_pb(hsv_img):
    hsv_pb = hsv_img.copy()
    hsv_pb[:, :, 0] = 0
    hsv_pb[:, :, 1] = 0
    return hsv_pb