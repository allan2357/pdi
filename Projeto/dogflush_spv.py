# ============================================================
# MCZA018 – Processamento Digital de Imagens
# DogFlush – Sistema de Descarga Automática para Pets (v4)
#
# Integrantes:
#   - [NOME COMPLETO 1] — RA: [RA 1]
#   - [NOME COMPLETO 2] — RA: [RA 2]
#   - [NOME COMPLETO 3] — RA: [RA 3]
#
# Data: Março de 2026
# Nome do programa: DogFlush SPV v4
#
# Chamada Linux:
#   python3 dogflush_spv.py
#   python3 dogflush_spv.py --entrada video.mp4
# ============================================================

import cv2
import numpy as np
import time
import argparse
import os


def selecionar_roi(frame):
    """Permite ao usuário selecionar a Região de Interesse (ROI) na tela."""
    instrucao = frame.copy()
    cv2.putText(instrucao, 'Selecione a area do pet e pressione ENTER', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(instrucao, 'Pressione C para cancelar (usa frame inteiro)', (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    roi = cv2.selectROI('DogFlush - Selecionar ROI', instrucao, showCrosshair=True)
    cv2.destroyWindow('DogFlush - Selecionar ROI')
    if roi[2] == 0 or roi[3] == 0:
        return None
    return roi


def executar_sistema_dogflush(entrada=0, gravar_video=True, caminho_saida="dogflush_saida.avi"):
    """
    Sistema de Processamento Visual DogFlush (v4 - Fusão OR).

    Arquitetura:
        1. Captura & Suavização (Gaussiano)
        2. Pipeline de Movimento (MOG2)
        3. Pipeline Estático de Cor (HSV Saturação + Otsu)
        4. Fusão Lógica (OR) — nunca perde rastreamento
        5. Limpeza Morfológica (Abertura + Fechamento)
        6. Watershed (sementes topográficas)

    Parâmetros:
        entrada: 0 para webcam, ou caminho de arquivo de vídeo (string)
        gravar_video: se True, grava a saída processada em arquivo
        caminho_saida: nome do arquivo de vídeo de saída
    """
    print(f"=== DogFlush v4 (Fusão OR) ===")
    print(f"Entrada: {'Webcam' if entrada == 0 else entrada}")
    print("Pressione 'ESC' para encerrar, 'R' para resetar o fundo.")
    print("-" * 50)

    cap = cv2.VideoCapture(entrada)
    if not cap.isOpened():
        print(f"Erro: Não foi possível abrir a fonte de vídeo: {entrada}")
        return None

    if entrada == 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
        cap.set(cv2.CAP_PROP_FPS, 15)

    largura = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    altura = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_entrada = cap.get(cv2.CAP_PROP_FPS) or 15

    # Seleção de ROI
    ret, primeiro_frame = cap.read()
    if not ret:
        print("Erro ao capturar frame inicial.")
        return None
    primeiro_frame = cv2.resize(primeiro_frame, (640, 480))
    roi = selecionar_roi(primeiro_frame)

    if roi is not None:
        x_roi, y_roi, w_roi, h_roi = roi
        area_roi = w_roi * h_roi
        print(f"ROI selecionada: x={x_roi}, y={y_roi}, w={w_roi}, h={h_roi} ({area_roi} px²)")
    else:
        x_roi, y_roi, w_roi, h_roi = 0, 0, 640, 480
        area_roi = 640 * 480
        print("ROI: frame inteiro (nenhuma seleção feita)")

    gravador = None
    if gravar_video:
        codec = cv2.VideoWriter_fourcc(*'XVID')
        gravador = cv2.VideoWriter(caminho_saida, codec, fps_entrada, (640, 480))
        print(f"Gravando vídeo de saída em: {caminho_saida}")

    # --- PIPELINE DE MOVIMENTO: MOG2 ---
    fgbg = cv2.createBackgroundSubtractorMOG2(
        history=500, varThreshold=50, detectShadows=True
    )
    taxa_aprendizado = 0.0005

    # Máquina de Estados: 0=Livre, 1=Ocupado, 2=Acionando
    ESTADO = 0
    frames_ocupado = 0
    LIMITE_FRAMES = 15
    total_ativacoes = 0

    # Kernels morfológicos
    kernel_abertura = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_fechamento = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    kernel_dilatacao = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_grande = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))

    # Buffer temporal (mediana de áreas)
    TAMANHO_BUFFER = 10
    buffer_areas = [0.0] * TAMANHO_BUFFER
    indice_buffer = 0

    # Limiares de filtragem por contorno
    PERCENTUAL_AREA_MIN = 0.03
    PERCENTUAL_AREA_MAX = 0.90
    ASPECT_RATIO_MIN = 0.2
    ASPECT_RATIO_MAX = 5.0
    SOLIDEZ_MINIMA = 0.1
    MARGEM_BORDA = 3

    # Métricas de desempenho
    metricas = {
        "total_frames": 0,
        "frames_com_deteccao": 0,
        "total_ativacoes": 0,
        "tempos_fps": [],
        "tempos_resposta": [],
        "inicio_ocupacao": None,
        "areas_detectadas": [],
        "contornos_rejeitados": {"area": 0, "forma": 0, "solidez": 0, "borda": 0}
    }

    # Memória temporal: a máscara "desbota" em vez de sumir instantaneamente
    FATOR_MEMORIA = 0.90  # 90% da detecção anterior é mantida
    mascara_memoria = None

    tempo_inicio_global = time.time()

    while True:
        tempo_frame = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))
        metricas["total_frames"] += 1

        # Recorta a ROI
        regiao = frame[y_roi:y_roi+h_roi, x_roi:x_roi+w_roi]

        # ============================================================
        # ETAPA 1: CAPTURA & SUAVIZAÇÃO (Aula 4)
        # Gaussiano remove ruídos de alta frequência
        # ============================================================
        regiao_suavizada = cv2.GaussianBlur(regiao, (7, 7), 0)

        # ============================================================
        # ETAPA 2: PROCESSAMENTO DE CORES (Aulas 7 e 10)
        # HSV para pipeline estático, Cinza para CLAHE
        # ============================================================
        hsv = cv2.cvtColor(regiao_suavizada, cv2.COLOR_BGR2HSV)
        cinza = cv2.cvtColor(regiao_suavizada, cv2.COLOR_BGR2GRAY)
        canal_s = hsv[:, :, 1]

        # ============================================================
        # ETAPA 3: EQUALIZAÇÃO DE HISTOGRAMA (CLAHE)
        # ============================================================
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        equalizado = clahe.apply(cinza)

        # Histograma comparativo (visualização didática)
        hist_orig = cv2.calcHist([cinza], [0], None, [256], [0, 256])
        hist_eq = cv2.calcHist([equalizado], [0], None, [256], [0, 256])
        img_hist = np.zeros((200, 512, 3), dtype=np.uint8)
        cv2.normalize(hist_orig, hist_orig, 0, 190, cv2.NORM_MINMAX)
        cv2.normalize(hist_eq, hist_eq, 0, 190, cv2.NORM_MINMAX)
        for i in range(256):
            cv2.line(img_hist, (i*2, 200), (i*2, 200 - int(hist_orig[i][0])), (100, 100, 255), 1)
            cv2.line(img_hist, (i*2+1, 200), (i*2+1, 200 - int(hist_eq[i][0])), (50, 255, 50), 1)
        cv2.putText(img_hist, "Vermelho=Original  Verde=CLAHE", (10, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        # ============================================================
        # ETAPA 4a: PIPELINE DE MOVIMENTO (MOG2)
        # Rastreia pixels em movimento. Se o cachorro deitar,
        # essa máscara vai sumir gradualmente.
        # ============================================================
        mascara_mog2 = fgbg.apply(regiao_suavizada, learningRate=taxa_aprendizado)
        mascara_mov = np.where(mascara_mog2 == 255, 255, 0).astype(np.uint8)

        # ============================================================
        # ETAPA 4b: PIPELINE ESTÁTICO DE COR (HSV Saturação + Otsu)
        # Extrai canal S e aplica Otsu. Cria máscara binária estática
        # baseada no contraste da pelagem. Funciona mesmo com o
        # cachorro totalmente parado.
        # ============================================================
        _, mascara_cor = cv2.threshold(canal_s, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # ============================================================
        # ETAPA 4c: FUSÃO LÓGICA — OR (O "Pulo do Gato")
        #
        # Se o cachorro estiver se movendo → MOG2 detecta
        # Se o cachorro deitar e parar   → Otsu/HSV detecta pela cor
        #
        # A união garante que o sistema NUNCA perca o rastreamento.
        # ============================================================
        mascara_fundida = cv2.bitwise_or(mascara_mov, mascara_cor)

        # ============================================================
        # ETAPA 5: LIMPEZA MORFOLÓGICA (Aula 8)
        # Abertura remove ruídos pequenos, Fechamento preenche buracos
        # ============================================================
        etapa1 = cv2.morphologyEx(mascara_fundida, cv2.MORPH_OPEN, kernel_abertura, iterations=2)
        etapa2 = cv2.morphologyEx(etapa1, cv2.MORPH_CLOSE, kernel_fechamento, iterations=3)
        etapa3 = cv2.dilate(etapa2, kernel_dilatacao, iterations=1)

        # Memória temporal com decaimento exponencial:
        # Mesmo que a detecção caia por alguns frames, a silhueta
        # anterior permanece e "desbota" gradualmente
        if mascara_memoria is None:
            mascara_memoria = etapa3.astype(np.float32)
        else:
            mascara_memoria = mascara_memoria * FATOR_MEMORIA + etapa3.astype(np.float32) * (1.0 - FATOR_MEMORIA)
        mascara_final = (mascara_memoria > 40).astype(np.uint8) * 255

        # ============================================================
        # ETAPA 6: WATERSHED (Aula 11)
        # Transformada de distância gera sementes topográficas
        # ============================================================
        fundo_certo = cv2.dilate(mascara_final, kernel_grande, iterations=2)
        dist_transform = cv2.distanceTransform(mascara_final, cv2.DIST_L2, 5)
        max_dist = dist_transform.max()

        frame_resultado = frame.copy()
        cv2.rectangle(frame_resultado, (x_roi, y_roi), (x_roi+w_roi, y_roi+h_roi), (255, 255, 0), 1)

        regiao_resultado = regiao.copy()
        if max_dist > 0:
            _, frente_certa = cv2.threshold(
                dist_transform, 0.4 * max_dist, 255, 0
            )
            frente_certa = np.uint8(frente_certa)
            marcadores = cv2.connectedComponents(frente_certa)[1] + 1
            regiao_desconhecida = cv2.subtract(fundo_certo, frente_certa)
            marcadores[regiao_desconhecida == 255] = 0
            cv2.watershed(regiao_resultado, marcadores)
            regiao_resultado[marcadores == -1] = [0, 0, 255]
        frame_resultado[y_roi:y_roi+h_roi, x_roi:x_roi+w_roi] = regiao_resultado

        # ============================================================
        # ETAPA 7: EXTRAÇÃO + FILTRAGEM POR FORMA
        # ============================================================
        contornos, _ = cv2.findContours(
            mascara_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        area_min_abs = int(area_roi * PERCENTUAL_AREA_MIN)
        area_max_abs = int(area_roi * PERCENTUAL_AREA_MAX)

        teve_movimento = False
        melhor_contorno = None
        melhor_area = 0

        for cnt in contornos:
            area = cv2.contourArea(cnt)
            x, y, w, h = cv2.boundingRect(cnt)

            # Filtro 0: Exclusão de bordas da ROI
            if (x <= MARGEM_BORDA or y <= MARGEM_BORDA or
                x + w >= w_roi - MARGEM_BORDA or y + h >= h_roi - MARGEM_BORDA):
                metricas["contornos_rejeitados"]["borda"] += 1
                continue

            # Filtro 1: Área (percentual da ROI)
            if area < area_min_abs or area > area_max_abs:
                metricas["contornos_rejeitados"]["area"] += 1
                continue

            # Filtro 2: Aspect Ratio
            if h == 0:
                continue
            aspect_ratio = w / h
            if aspect_ratio < ASPECT_RATIO_MIN or aspect_ratio > ASPECT_RATIO_MAX:
                metricas["contornos_rejeitados"]["forma"] += 1
                continue

            # Filtro 3: Solidez
            hull = cv2.convexHull(cnt)
            area_hull = cv2.contourArea(hull)
            if area_hull > 0:
                solidez = area / area_hull
                if solidez < SOLIDEZ_MINIMA:
                    metricas["contornos_rejeitados"]["solidez"] += 1
                    continue

            if area > melhor_area:
                melhor_area = area
                melhor_contorno = cnt

        if melhor_contorno is not None:
            teve_movimento = True
            metricas["frames_com_deteccao"] += 1
            metricas["areas_detectadas"].append(melhor_area)

            x, y, w, h = cv2.boundingRect(melhor_contorno)
            aspect_ratio = w / h
            hull = cv2.convexHull(melhor_contorno)
            solidez = melhor_area / max(cv2.contourArea(hull), 1)

            # Coordenadas absolutas (ROI → frame completo)
            xa, ya = x + x_roi, y + y_roi
            cv2.rectangle(frame_resultado, (xa, ya), (xa+w, ya+h), (0, 255, 0), 2)
            cnt_absoluto = melhor_contorno.copy()
            cnt_absoluto[:, :, 0] += x_roi
            cnt_absoluto[:, :, 1] += y_roi
            cv2.drawContours(frame_resultado, [cnt_absoluto], -1, (0, 255, 100), 1)

            pct_roi = (melhor_area / area_roi) * 100
            info = f"A:{int(melhor_area)} ({pct_roi:.0f}%ROI) S:{solidez:.2f}"
            cv2.putText(frame_resultado, info,
                        (xa, ya - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # Suavização temporal com mediana
        buffer_areas[indice_buffer] = melhor_area if teve_movimento else 0.0
        indice_buffer = (indice_buffer + 1) % TAMANHO_BUFFER
        mediana_areas = float(np.median(buffer_areas))
        deteccao_suavizada = mediana_areas >= area_min_abs

        # ============================================================
        # MÁQUINA DE ESTADOS TEMPORAL
        # ============================================================
        if ESTADO == 0:
            msg, cor = "SISTEMA: MONITORANDO (LIVRE)", (255, 100, 0)
            if deteccao_suavizada:
                frames_ocupado += 1
                if frames_ocupado > LIMITE_FRAMES:
                    ESTADO = 1
                    metricas["inicio_ocupacao"] = time.time()
            else:
                frames_ocupado = max(0, frames_ocupado - 1)

        elif ESTADO == 1:
            msg, cor = "SISTEMA: USO EM ANDAMENTO", (0, 165, 255)
            if deteccao_suavizada:
                # Reforça a permanência no estado ocupado
                frames_ocupado = min(frames_ocupado + 1, LIMITE_FRAMES * 3)
            else:
                frames_ocupado -= 1
                if frames_ocupado <= 0:
                    ESTADO = 2

        elif ESTADO == 2:
            msg, cor = "!!! DESCARGA ATIVADA !!!", (0, 0, 255)
            total_ativacoes += 1
            metricas["total_ativacoes"] = total_ativacoes

            if metricas["inicio_ocupacao"]:
                tempo_resp = time.time() - metricas["inicio_ocupacao"]
                metricas["tempos_resposta"].append(tempo_resp)
                metricas["inicio_ocupacao"] = None

            cv2.imshow('DogFlush - 8. Resultado Final', frame_resultado)
            cv2.waitKey(2000)
            ESTADO, frames_ocupado = 0, 0
            buffer_areas = [0.0] * TAMANHO_BUFFER
            mascara_memoria = None
            fgbg = cv2.createBackgroundSubtractorMOG2(
                history=500, varThreshold=50, detectShadows=True
            )

        # FPS
        fps_atual = 1.0 / max(time.time() - tempo_frame, 0.001)
        metricas["tempos_fps"].append(fps_atual)

        # HUD
        cv2.putText(frame_resultado, msg, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, cor, 2)
        cv2.putText(frame_resultado, f"Descargas: {total_ativacoes}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        cv2.putText(frame_resultado, f"FPS: {fps_atual:.1f}",
                    (540, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(frame_resultado, f"Frame: {metricas['total_frames']}",
                    (540, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        # Barra de progresso do estado
        barra_y = 470
        barra_w = int((frames_ocupado / max(LIMITE_FRAMES, 1)) * 200)
        cv2.rectangle(frame_resultado, (10, barra_y), (210, barra_y + 8), (50, 50, 50), -1)
        cv2.rectangle(frame_resultado, (10, barra_y), (10 + min(barra_w, 200), barra_y + 8), cor, -1)

        # Visualização HSV (saturação com colormap)
        vis_hsv = cv2.applyColorMap(canal_s, cv2.COLORMAP_JET)

        # Colorir as máscaras para visualização didática
        vis_mov = cv2.cvtColor(mascara_mov, cv2.COLOR_GRAY2BGR)
        vis_mov[mascara_mov == 255] = [255, 100, 0]  # azul = movimento

        vis_cor = cv2.cvtColor(mascara_cor, cv2.COLOR_GRAY2BGR)
        vis_cor[mascara_cor == 255] = [0, 200, 255]   # amarelo = cor estática

        vis_fundida = cv2.cvtColor(mascara_final, cv2.COLOR_GRAY2BGR)
        vis_fundida[mascara_final == 255] = [0, 255, 0]  # verde = fusão final

        # ============================================================
        # EXIBIÇÃO DIDÁTICA — 8 janelas
        # ============================================================
        cv2.imshow('DogFlush - 1. Blur Gaussiano', cv2.resize(regiao_suavizada, (320, 240)))
        cv2.imshow('DogFlush - 2. CLAHE', cv2.resize(equalizado, (320, 240)))
        cv2.imshow('DogFlush - 3. Histograma', img_hist)
        cv2.imshow('DogFlush - 4. Saturacao HSV', cv2.resize(vis_hsv, (320, 240)))
        cv2.imshow('DogFlush - 5. MOG2 (Movimento)', cv2.resize(vis_mov, (320, 240)))
        cv2.imshow('DogFlush - 6. Otsu/HSV (Cor)', cv2.resize(vis_cor, (320, 240)))
        cv2.imshow('DogFlush - 7. Mascara Fundida (OR)', cv2.resize(vis_fundida, (320, 240)))
        cv2.imshow('DogFlush - 8. Resultado Final', frame_resultado)

        if gravador:
            gravador.write(frame_resultado)

        tecla = cv2.waitKey(30) & 0xFF
        if tecla == 27:
            break
        elif tecla == ord('r') or tecla == ord('R'):
            fgbg = cv2.createBackgroundSubtractorMOG2(
                history=500, varThreshold=50, detectShadows=True
            )
            buffer_areas = [0.0] * TAMANHO_BUFFER
            mascara_memoria = None
            print(">>> Modelo de fundo resetado!")

    # Finalização
    tempo_total = time.time() - tempo_inicio_global
    cap.release()
    if gravador:
        gravador.release()
        print(f"\nVídeo salvo em: {caminho_saida}")
    cv2.destroyAllWindows()

    metricas["tempo_total_seg"] = tempo_total
    metricas["fps_medio"] = np.mean(metricas["tempos_fps"]) if metricas["tempos_fps"] else 0
    metricas["taxa_deteccao"] = (
        metricas["frames_com_deteccao"] / max(metricas["total_frames"], 1)
    ) * 100
    metricas["area_media"] = np.mean(metricas["areas_detectadas"]) if metricas["areas_detectadas"] else 0
    metricas["tempo_resposta_medio"] = (
        np.mean(metricas["tempos_resposta"]) if metricas["tempos_resposta"] else 0
    )

    print("\n" + "=" * 55)
    print("      RELATÓRIO DE MÉTRICAS DE DESEMPENHO (v4)")
    print("=" * 55)
    print(f"  Tempo total de execução:    {tempo_total:.1f}s")
    print(f"  Total de frames:            {metricas['total_frames']}")
    print(f"  FPS médio:                  {metricas['fps_medio']:.1f}")
    print(f"  Frames com detecção:        {metricas['frames_com_deteccao']}")
    print(f"  Taxa de detecção:           {metricas['taxa_deteccao']:.1f}%")
    print(f"  Área média detectada:       {metricas['area_media']:.0f} px²")
    print(f"  Total de ativações:         {metricas['total_ativacoes']}")
    print(f"  Tempo resp. médio:          {metricas['tempo_resposta_medio']:.2f}s")
    print(f"  --- Contornos rejeitados ---")
    print(f"  Por borda (toca borda):      {metricas['contornos_rejeitados']['borda']}")
    print(f"  Por área (muito grande/peq): {metricas['contornos_rejeitados']['area']}")
    print(f"  Por forma (aspect ratio):    {metricas['contornos_rejeitados']['forma']}")
    print(f"  Por solidez (irregular):     {metricas['contornos_rejeitados']['solidez']}")
    print("=" * 55)

    return metricas


def calcular_psnr(img_original, img_processada):
    """Calcula o PSNR entre duas imagens."""
    mse = np.mean((img_original.astype(float) - img_processada.astype(float)) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10((255.0 ** 2) / mse)


def avaliar_qualidade_filtragem(caminho_video=0):
    """Avalia a qualidade da filtragem via PSNR."""
    cap = cv2.VideoCapture(caminho_video)
    if not cap.isOpened():
        print("Erro ao abrir vídeo.")
        return

    valores_psnr = []
    for i in range(50):
        ret, frame = cap.read()
        if not ret:
            break
        cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        suavizado = cv2.GaussianBlur(cinza, (7, 7), 0)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        equalizado = clahe.apply(suavizado)
        psnr = calcular_psnr(cinza, equalizado)
        valores_psnr.append(psnr)

    cap.release()
    print("=== Análise de Qualidade Visual (PSNR) ===")
    print(f"  PSNR médio: {np.mean(valores_psnr):.2f} dB")
    print(f"  PSNR mínimo: {np.min(valores_psnr):.2f} dB")
    print(f"  PSNR máximo: {np.max(valores_psnr):.2f} dB")
    print(f"  Desvio padrão: {np.std(valores_psnr):.2f} dB")
    return valores_psnr


# Execução via linha de comando
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DogFlush SPV v4")
    parser.add_argument("--entrada", default=0, help="0 para webcam ou caminho do vídeo")
    parser.add_argument("--saida", default="dogflush_saida.avi", help="Caminho do vídeo de saída")
    parser.add_argument("--sem-gravar", action="store_true", help="Não gravar vídeo de saída")
    args = parser.parse_args()

    entrada = args.entrada if args.entrada != "0" else 0
    executar_sistema_dogflush(
        entrada=entrada,
        gravar_video=not args.sem_gravar,
        caminho_saida=args.saida
    )
