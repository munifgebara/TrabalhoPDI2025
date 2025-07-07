# 🖼️ Projeto de Processamento Digital de Imagens (PDI)

Este projeto realiza operações de processamento digital de imagens utilizando Python 3, com as bibliotecas OpenCV, NumPy e Matplotlib. Ele inclui funções para:

- Carregar e salvar imagens
- Conversão para HSV
- Normalização do canal V (brilho)
- Cálculo de limiar com o método de Otsu
- Correção gama com limiar adaptativo
- Transformação HSV → escala de cinza
- Visualização lado a lado

## 📁 Estrutura de pastas

```
projeto-pdi/
├── input/         # Imagens de entrada (.jpg)
├── output/        # Imagens processadas
├── utils_pdi.py   # Funções auxiliares
├── main.py        # Script principal de execução
├── requirements.txt
└── README.md
```

## 🛠️ Requisitos

- Python 3.8+
- NumPy
- OpenCV
- Matplotlib

Instale com:

```bash
pip install -r requirements.txt
```

## ▶️ Como executar

Execute o script principal:

```bash
python main.py
```

O script irá:

1. Processar todas as imagens `.jpg` na pasta `input/`
2. Aplicar transformação de brilho, correção gama ou outra operação conforme configurado
3. Salvar os resultados na pasta `output/`
4. Exibir visualização comparativa (original vs. processada)

## ⚙️ Funcionalidades implementadas

- ✅ Normalização de brilho (canal V)
- ✅ Correção gama adaptativa (dupla)
- ✅ Aplicação de limiar de Otsu
- ✅ Conversão para preto e branco via HSV
- ✅ Visualização com `matplotlib`

## 📌 Exemplos de uso

```python
from utils_pdi import normalizar_v_com_limite, corrigir_gama_duplo

# Normalizar pixels com V < limiar
img_hsv_norm = normalizar_v_com_limite(img_hsv, limite_v=80)

# Aplicar correção gama abaixo/acima do limiar de Otsu
img_hsv_corr = corrigir_gama_duplo(img_hsv, gama_baixo=2.0, gama_alto=0.8)
```

## 📷 Créditos

Projeto acadêmico de PDI — UEM / Pós-graduação em Ciência da Computação  
Autor: [Munif Gebara Junior](https://github.com/munifgebara)

---