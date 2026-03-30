import cv2
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    cv2.imshow("feed", frame)
    key = cv2.waitKey(1)

    if key & 0xFF == ord('q'):
        break
    elif key != -1:
        cv2.imwrite('pic.jpg', frame)


cv2.destroyAllWindows()