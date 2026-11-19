import os
import time
import warnings
import cv2
import joblib
import mediapipe as mp
import numpy as np
import pyautogui

# Ignora avisos desnecessários no terminal
warnings.filterwarnings("ignore")

# Configuração de segurança e delay do PyAutoGUI
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0.03  # Pequena pausa padrão entre comandos do PyAutoGUI

# Carrega o modelo treinado
MODEL_PATH = "hand_model.pkl"
TASK_PATH = "hand_landmarker.task"

if not os.path.exists(MODEL_PATH):
  raise FileNotFoundError(
      "Modelo 'hand_model.pkl' não encontrado. Execute o 'train_model.py'"
      " primeiro!"
  )

modelo = joblib.load(MODEL_PATH)
print("Modelo carregado com sucesso!")
print(f"Classes do modelo: {list(modelo.classes_)}")

if "apontar_esquerda" not in modelo.classes_ or "apontar_direita" not in modelo.classes_:
  print("\n[AVISO] O modelo atual ainda contem as classes antigas.")
  print("Lembre-se de rodar 'collect_data.py' e 'train_model.py' para registrar os novos gestos.\n")

# Configurações da Câmera (otimizadas para menor atraso)
CAMERA_INDEX = 0
camera = cv2.VideoCapture(CAMERA_INDEX)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not camera.isOpened():
  raise RuntimeError("Não foi possível abrir a webcam.")

# Configuração do MediaPipe Tasks na CPU
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=TASK_PATH),
    running_mode=RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)

# Conexões dos dedos para visualização
conexoes = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (0, 17),
]

DELAY_COMANDO = 0.12  # Intervalo mínimo (em segundos) entre ações
ultimo_comando_tempo = 0.0
tecla_atual = None

inicio = time.monotonic()
ultimo_timestamp = 0
gesto_anterior = None
texto_status = "Iniciando..."
texto_acao = "PARADO"

print("\nIniciando controle por gestos...")
print("Pressione 'Q' na janela do OpenCV para sair.")


def soltar_tecla_atual():
  global tecla_atual
  if tecla_atual is not None:
    pyautogui.keyUp(tecla_atual)
    tecla_atual = None
  pyautogui.keyUp("right")
  pyautogui.keyUp("left")


try:
  with HandLandmarker.create_from_options(options) as detector:
    while True:
      sucesso, frame = camera.read()
      if not sucesso or frame is None:
        break

      # Espelha a imagem para ficar natural
      frame = cv2.flip(frame, 1)

      # Conversão eficiente para o MediaPipe
      rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      imagem_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

      timestamp = int((time.monotonic() - inicio) * 1000)
      timestamp = max(timestamp, ultimo_timestamp + 1)
      ultimo_timestamp = timestamp

      resultado = detector.detect_for_video(imagem_mp, timestamp)

      if resultado.hand_landmarks:
        landmarks = resultado.hand_landmarks[0]
        altura, largura, _ = frame.shape

        # Desenha os pontos e conexões na tela
        pontos_tela = []
        for p in landmarks:
          px = int(p.x * largura)
          py = int(p.y * altura)
          pontos_tela.append((px, py))
          cv2.circle(frame, (px, py), 4, (0, 255, 0), -1)

        for a, b in conexoes:
          cv2.line(frame, pontos_tela[a], pontos_tela[b], (255, 0, 0), 2)

        # Normaliza os dados para o modelo
        pulso = landmarks[0]
        escala = max(
            max(abs(p.x - pulso.x), abs(p.y - pulso.y)) for p in landmarks
        )
        escala = max(escala, 0.0001)

        dados = []
        for p in landmarks:
          x = (p.x - pulso.x) / escala
          y = (p.y - pulso.y) / escala
          z = (p.z - pulso.z) / escala
          dados.extend([x, y, z])

        # Predição com o modelo treinado
        predicao = modelo.predict([dados])[0]
        probabilidades = modelo.predict_proba([dados])[0]
        confianca = max(probabilidades)

        # --- AÇÕES PARA O JOGO ---
        agora = time.monotonic()

        if confianca > 0.65:
          texto_status = f"Gesto: {predicao} ({confianca * 100:.1f}%)"

          # PULO: Mao fechada (disparo imediato ao fechar a mao)
          if predicao == "fechada":
            soltar_tecla_atual()
            if gesto_anterior != "fechada":
              # Envia 'space' (e 'up' para jogos tipo Subway Surfers)
              pyautogui.press("space")
              pyautogui.press("up")
              print(">>> PULO ACIONADO!")
            texto_acao = "PULO [Espaco / Up]"

          elif (agora - ultimo_comando_tempo) >= DELAY_COMANDO:
            # Movimento para DIREITA
            if predicao in ["apontar_direita", "dois_dedos"]:
              if tecla_atual != "right":
                pyautogui.keyUp("left")
                pyautogui.keyDown("right")
                tecla_atual = "right"
              ultimo_comando_tempo = agora
              texto_acao = "DIREITA [->]"

            # Movimento para ESQUERDA
            elif predicao in ["apontar_esquerda", "indicador"]:
              if tecla_atual != "left":
                pyautogui.keyUp("right")
                pyautogui.keyDown("left")
                tecla_atual = "left"
              ultimo_comando_tempo = agora
              texto_acao = "ESQUERDA [<-]"

            # NEUTRO / PARADO: Mao aberta
            elif predicao == "aberta":
              soltar_tecla_atual()
              texto_acao = "PARADO (Neutro)"

            else:
              soltar_tecla_atual()
              texto_acao = "PARADO"

          gesto_anterior = predicao
        else:
          texto_status = "Incerto..."
          soltar_tecla_atual()
          texto_acao = "PARADO"
      else:
        texto_status = "Nenhuma mao detectada"
        soltar_tecla_atual()
        texto_acao = "PARADO"

      # Exibe informações na janela do vídeo
      cv2.putText(
          frame,
          texto_status,
          (10, 35),
          cv2.FONT_HERSHEY_SIMPLEX,
          0.7,
          (0, 255, 0),
          2,
      )
      cv2.putText(
          frame,
          f"Acao: {texto_acao}",
          (10, 68),
          cv2.FONT_HERSHEY_SIMPLEX,
          0.65,
          (255, 255, 0),
          2,
      )
      cv2.imshow("Hand Controller - Jogo", frame)

      # Fecha se apertar 'q'
      if cv2.waitKey(1) & 0xFF == ord("q"):
        break
finally:
  soltar_tecla_atual()
  camera.release()
  cv2.destroyAllWindows()
  print("Encerrado.")