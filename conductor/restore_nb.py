import json
import os

def build_notebook():
    cells = []
    def add_md(text):
        cells.append({"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)})
    def add_code(text):
        cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text.splitlines(True)})

    add_md("""# MCZA018 – Processamento Digital de Imagens
## Relatório Final do Trabalho (RFT): DogFlush - Sistema de Descarga Automática para Pets

**Equipe:**
- NOME 1 - RA: 0000000
- NOME 2 - RA: 0000000
- NOME 3 - RA: 0000000

**Data de Publicação:** 29 de Março de 2026

---

## 1. Introdução

### Contexto e Cenário de Aplicação (CA)
O **DogFlush** é um Sistema de Processamento Visual (SPV) interativo criado para monitorar ambientes de higiene para animais de estimação.
O objetivo do sistema é analisar a cena visual em tempo real via webcam para detectar automaticamente o uso do banheirinho e acionar a limpeza.

### Fundamentação Teórica
O sistema utiliza conceitos de PDI como:
1. **Filtragem:** Gaussian Blur.
2. **Histograma:** CLAHE.
3. **Segmentação:** MOG2.
4. **Morfologia:** Opening/Closing.
5. **Avançado:** Watershed.

---

## 2. Materiais e Métodos
Pipeline: `Captura` -> `Grayscale` -> `Blur` -> `CLAHE` -> `MOG2` -> `Morfologia` -> `Watershed` -> `Máquina de Estados`.
""")

    code_text = """# CABEÇALHO OBRIGATÓRIO
# Nome da Equipe: [Inserir Nome]
# Nome do Programa: dog_flush_spv.py

import cv2
import numpy as np
import time

def executar_sistema_dogflush(modo_video=0):
    print("Iniciando Módulo de Câmera... Pressione 'ESC' para encerrar.")
    
    cap = cv2.VideoCapture(modo_video)
    if not cap.isOpened(): return

    # Padronizando a câmera: Resolução (640x480) e FPS (15)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 15)

    fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

    ESTADO = 0 
    frames_ocupado = 0
    LIMITE_FRAMES = 15 
    total_ativacoes = 0
    kernel = np.ones((5,5), np.uint8)

    while True:
        ret, frame = cap.read()
        if not ret: break
            
        frame = cv2.resize(frame, (640, 480))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (7, 7), 0)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        eq = clahe.apply(blur)
        
        fgmask = fgbg.apply(eq)
        _, fgmask_bin = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)
        opening = cv2.morphologyEx(fgmask_bin, cv2.MORPH_OPEN, kernel, iterations=1)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Watershed
        sure_bg = cv2.dilate(closing, kernel, iterations=3)
        dist_transform = cv2.distanceTransform(closing, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        markers = cv2.connectedComponents(sure_fg)[1] + 1
        markers[cv2.subtract(sure_bg, sure_fg) == 255] = 0
        frame_watershed = frame.copy()
        cv2.watershed(frame_watershed, markers)
        frame_watershed[markers == -1] = [0, 0, 255]
        
        # Máquina de Estado
        contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        teve_movimento = False
        for cnt in contours:
            if cv2.contourArea(cnt) > 8000: 
                teve_movimento = True
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(frame_watershed, (x, y), (x+w, y+h), (0, 255, 0), 2)
                break
                
        if ESTADO == 0:
            msg, cor = "SISTEMA: MONITORANDO (LIVRE)", (255,0,0)
            if teve_movimento:
                frames_ocupado += 1
                if frames_ocupado > LIMITE_FRAMES: ESTADO = 1
        elif ESTADO == 1:
            msg, cor = "SISTEMA: USO EM ANDAMENTO", (0,165,255)
            if not teve_movimento:
                frames_ocupado -= 1
                if frames_ocupado <= 0: ESTADO = 2
        elif ESTADO == 2:
            msg, cor = "!!! DESCARGA ATIVADA !!!", (0,0,255)
            total_ativacoes += 1
            cv2.imshow('5. DogFlush Final', frame_watershed)
            cv2.waitKey(2000)
            ESTADO, frames_ocupado = 0, 0
            fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)
            
        cv2.putText(frame_watershed, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, cor, 2)
        cv2.putText(frame_watershed, f"Descargas: {total_ativacoes}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)

        # Exibição Didática
        cv2.imshow('1. Grayscale + Blur', cv2.resize(blur, (320, 240)))
        cv2.imshow('2. Equalizacao (CLAHE)', cv2.resize(eq, (320, 240)))
        cv2.imshow('3. Segmentacao (MOG2)', cv2.resize(fgmask, (320, 240)))
        cv2.imshow('4. Morfologia', cv2.resize(closing, (320, 240)))
        cv2.imshow('5. DogFlush Final', frame_watershed)

        if cv2.waitKey(30) & 0xFF == 27: break

    cap.release()
    cv2.destroyAllWindows()

# executar_sistema_dogflush(0)
"""
    add_code(code_text)
    add_md("""---
## 3. Laboratório Experimental (LEx) e Conclusões...""")

    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3","name": "python3"}}, "nbformat": 4, "nbformat_minor": 4}
    with open(r"C:\Users\allan\Downloads\pdi\Projeto\DogFlush_Entregavel.ipynb", 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    build_notebook()
