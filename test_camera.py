
import cv2

for i in range(10):

    camera = cv2.VideoCapture(i)

    if camera.isOpened():

        sucesso, frame = camera.read()

        if sucesso:
            print(f"Camera encontrada: {i}")

            cv2.imshow(f"Camera {i}", frame)
            cv2.waitKey(1500)

        camera.release()

cv2.destroyAllWindows()
