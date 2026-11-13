import os
import time
import cv2
import mediapipe as mp
import pandas as pd

# CONFIGURACOES
CAMERA_INDEX = 0
MODEL_PATH = "hand_landmarker.task"
DATASET_PATH = "dataset/gestures.csv"

INTERVALO = 0.15

GESTOS = {
    ord("1"): "aberta",
    ord("2"): "fechada",
    ord("3"): "indicador",
    ord("4"): "dois_dedos",
}

os.makedirs("dataset", exist_ok=True)

# COLUNAS DO CSV
colunas = ["label"]

for i in range(21):
  colunas.extend([f"x{i}", f"y{i}", f"z{i}"])

# VERIFICA MODELO
if not os.path.exists(MODEL_PATH):
  raise FileNotFoundError("Arquivo hand_landmarker.task nao encontrado!")

# CONFIGURA MEDIAPIPE TASKS
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.6,
    min_hand_presence_confidence=0.6,
    min_tracking_confidence=0.6,
)

# ABRE CAMERA VIA STREAM IP DO DROIDCAM
# (Substitua 192.168.X.X pelo IP que aparece no app do seu celular)
url = "http://192.168.0.2:4747/video"
camera = cv2.VideoCapture(url)

if not camera.isOpened():
  raise RuntimeError(
      "Nao foi possivel conectar ao DroidCam via IP. "
      "Verifique se o IP esta correto, se o celular e o PC estao "
      "na mesma rede Wi-Fi e se o app DroidCam esta aberto na tela."
  )
if not camera.isOpened():
  raise RuntimeError(
      "Nao foi possivel abrir a camera. Verifique o indice do DroidCam."
  )

# VARIAVEIS
gesto_ativo = None
ultimo_registro = 0
inicio = time.monotonic()
ultimo_timestamp = 0

# CONEXOES DOS DEDOS
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

print("COLETA DE DADOS INICIADA")
print("1 - Mao aberta")
print("2 - Punho fechado")
print("3 - Indicador")
print("4 - Dois dedos")
print("ESPACO - Pausar coleta")
print("Q - Sair")

with HandLandmarker.create_from_options(options) as detector:

  while True:
    sucesso, frame = camera.read()

    if not sucesso or frame is None:
      print("Erro ao capturar imagem.")
      break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    imagem_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    timestamp = int((time.monotonic() - inicio) * 1000)

    timestamp = max(timestamp, ultimo_timestamp + 1)
    ultimo_timestamp = timestamp

    resultado = detector.detect_for_video(imagem_mp, timestamp)

    tecla = cv2.waitKey(1) & 0xFF

    if tecla in GESTOS:
      gesto_ativo = GESTOS[tecla]

    elif tecla == ord(" "):
      gesto_ativo = None

    if resultado.hand_landmarks:
      landmarks = resultado.hand_landmarks[0]

      # DESENHA OS PONTOS
      altura, largura, _ = frame.shape

      pontos_tela = []

      for p in landmarks:
        px = int(p.x * largura)
        py = int(p.y * altura)

        pontos_tela.append((px, py))

        cv2.circle(frame, (px, py), 5, (0, 255, 0), -1)

      # DESENHA CONEXOES
      for a, b in conexoes:
        cv2.line(frame, pontos_tela[a], pontos_tela[b], (255, 0, 0), 2)

      # NORMALIZA COORDENADAS
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

      # COLETA
      agora = time.monotonic()

      if gesto_ativo and agora - ultimo_registro >= INTERVALO:
        linha = [gesto_ativo] + dados

        arquivo_existe = os.path.exists(DATASET_PATH) and (
            os.path.getsize(DATASET_PATH) > 0
        )

        df = pd.DataFrame([linha], columns=colunas)

        df.to_csv(
            DATASET_PATH,
            mode="a",
            header=not arquivo_existe,
            index=False,
        )

        ultimo_registro = agora

        print("Amostra:", gesto_ativo)

    # INFORMACOES NA TELA
    status = gesto_ativo if gesto_ativo else "PAUSADO"

    cv2.putText(
        frame,
        f"Gesto: {status}",
        (10, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
    )

    cv2.putText(
        frame,
        "1-4: Selecionar | ESPACO: Pausar | Q: Sair",
        (10, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        2,
    )

    cv2.imshow("Hand Controller - Coleta", frame)

    if tecla == ord("q"):
      break

camera.release()
cv2.destroyAllWindows()

print("Coleta finalizada!")