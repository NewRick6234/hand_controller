
import cv2

backends = [
    ("DirectShow", cv2.CAP_DSHOW),
    ("Media Foundation", cv2.CAP_MSMF),
    ("Automatico", cv2.CAP_ANY)
]

for nome_backend, backend in backends:

    for indice in range(5):

        print(f"\nTestando {nome_backend} - Camera {indice}")

        camera = cv2.VideoCapture(indice, backend)

        if not camera.isOpened():
            print("Camera indisponivel")
            camera.release()
            continue

        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        for _ in range(10):
            sucesso, frame = camera.read()

        if sucesso and frame is not None:
            print("Imagem recebida:", frame.shape)

            cv2.imshow(
                f"{nome_backend} - Camera {indice}",
                frame
            )

            print("Pressione Q para testar a proxima camera.")

            while True:
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        else:
            print("Nao foi possivel receber imagem.")

        camera.release()
        cv2.destroyAllWindows()

print("\nTestes finalizados.")
