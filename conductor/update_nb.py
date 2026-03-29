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

---

## 1. Introdução e Evolução do Projeto
O **DogFlush** evoluiu de um sistema de detecção de movimento puro para uma **Abordagem Híbrida de Segmentação**. 
Pelas análises de campo, notou-se que o cachorro, ao ficar estático, era absorvido pelo fundo nos algoritmos tradicionais de vídeo (MOG2). Para solucionar isso, implementamos técnicas avançadas de PDI:
1. **Segmentação por Cor (Espaço HSV):** Isolamento cromático do animal em relação ao piso.
2. **Subtração de Fundo Estática:** Comparação com um frame de referência fixo.
3. **Filtro Bilateral:** Suavização seletiva que preserva as bordas da silhueta.

---

## 2. Materiais e Métodos (Pipeline Híbrido)
`Webcam` -> `Filtro Bilateral` -> `Conversão HSV` -> `Máscara de Cor` + `Diferença de Fundo Estático` -> `Morfologia (Opening/Closing)` -> `Watershed` -> `Máquina de Estados`.
""")

    code_text = """# CABEÇALHO OBRIGATÓRIO
# Nome da Equipe: [Inserir Nome]
# Nome do Programa: dog_flush_v2_hibrido.py

import cv2
import numpy as np
import time

def executar_sistema_dogflush(modo_video=0):
    print("Iniciando Sistema Híbrido...")
    print("COMANDOS: 'S' para Capturar Fundo Vazio | 'ESC' para Sair")
    
    cap = cv2.VideoCapture(modo_video)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 15)

    background_ref = None
    total_ativacoes = 0
    ESTADO = 0 # 0=Livre, 1=Ocupado, 2=Acionando
    frames_estatado = 0
    
    # Kernel para Morfologia
    kernel = np.ones((7,7), np.uint8)

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.resize(frame, (640, 480))
        # 1. FILTRAGEM: Bilateral Filter (Preserva bordas melhor que o Gaussian)
        blur = cv2.bilateralFilter(frame, 9, 75, 75)
        
        # 2. PROCESSAMENTO DE CORES (HSV)
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
        # Máscara para detectar cores escuras/marrons (Cachorro) vs Chão Claro
        # Ajustamos para pegar qualquer coisa que NÃO seja o branco/bege do chão
        lower_dog = np.array([0, 0, 0])
        upper_dog = np.array([180, 255, 130]) # Foco em Luminância Baixa (Objetos escuros)
        mask_color = cv2.inRange(hsv, lower_dog, upper_dog)

        # 3. SUBTRAÇÃO DE FUNDO ESTÁTICA (Melhor que MOG2 para objetos parados)
        mask_diff = np.zeros((480, 640), dtype=np.uint8)
        if background_ref is not None:
            diff = cv2.absdiff(cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY), background_ref)
            _, mask_diff = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

        # 4. COMBINAÇÃO HÍBRIDA (Cor OR Diferença)
        combined_mask = cv2.bitwise_or(mask_color, mask_diff)
        
        # 5. MORFOLOGIA
        opening = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=3)
        
        # 6. WATERSHED (Bordas Vermelhas)
        dist_transform = cv2.distanceTransform(closing, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.4 * dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        markers = cv2.connectedComponents(sure_fg)[1] + 1
        markers[cv2.subtract(cv2.dilate(closing, kernel, iterations=3), sure_fg) == 255] = 0
        frame_out = frame.copy()
        cv2.watershed(frame_out, markers)
        frame_out[markers == -1] = [0, 0, 255]

        # 7. MÁQUINA DE ESTADOS
        contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        teve_alvo = False
        for cnt in contours:
            if cv2.contourArea(cnt) > 12000: # Ajustado para o tamanho do cão na foto
                teve_alvo = True
                x,y,w,h = cv2.boundingRect(cnt)
                cv2.rectangle(frame_out, (x,y), (x+w, y+h), (0,255,0), 2)
                break

        # Lógica de transição
        if ESTADO == 0:
            status_txt = "STATUS: MONITORANDO (LIVRE)"
            color_txt = (255,0,0)
            if teve_alvo:
                frames_estatado += 1
                if frames_estatado > 10: ESTADO = 1
        elif ESTADO == 1:
            status_txt = "STATUS: CÃO DETECTADO (OCUPADO)"
            color_txt = (0,165,255)
            if not teve_alvo:
                frames_estatado -= 1
                if frames_estatado <= 0: ESTADO = 2
        elif ESTADO == 2:
            status_txt = "!!! ACIONANDO DESCARGA !!!"
            color_txt = (0,0,255)
            total_ativacoes += 1
            cv2.imshow('DogFlush Final', frame_out)
            cv2.waitKey(2000)
            ESTADO = 0
            frames_estatado = 0

        # UI e Exibição
        cv2.putText(frame_out, status_txt, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_txt, 2)
        cv2.putText(frame_out, f"Descargas: {total_ativacoes}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
        
        if background_ref is None:
            cv2.putText(frame_out, "APERTE 'S' PARA CALIBRAR FUNDO", (150, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

        cv2.imshow('DogFlush Final', frame_out)
        cv2.imshow('Debug: Mascara Hibrida', closing)

        key = cv2.waitKey(30) & 0xFF
        if key == 27: break
        elif key == ord('s'):
            background_ref = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
            print("Fundo capturado com sucesso!")

    cap.release()
    cv2.destroyAllWindows()

# executar_sistema_dogflush(0)
"""
    add_code(code_text)
    
    add_md("""---
## 3. Conclusões da Melhoria
A substituição do MOG2 pela combinação de **Máscara HSV + Diferença de Fundo Estática** resolveu o problema do animal ser "esquecido" pelo sistema ao dormir/ficar parado. O Filtro Bilateral garantiu que o Watershed não criasse contornos espúrios dentro da pelagem do animal, focando apenas na silhueta externa.""")

    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3","name": "python3"}}, "nbformat": 4, "nbformat_minor": 4}
    with open(r"C:\Users\allan\Downloads\pdi\Projeto\DogFlush_Entregavel.ipynb", 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    build_notebook()
