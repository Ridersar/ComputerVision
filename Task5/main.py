import cv2

# проход по всем изображениям
for number in range(1, 4):
    image = cv2.imread("resources/{}.jpg".format(number))

    # приведение изображения к ЧБ
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # распознаватель лиц (каскад Хаара)
    face_cascade = cv2.CascadeClassifier("resources/haar.xml")

    # обнаружение лиц на изображении
    faces = face_cascade.detectMultiScale(image_gray, minSize=(70, 70), minNeighbors=12)

    print("Фотография №{}. Обнаружено лиц:".format(number), len(faces))

    # обводка обнаруженных лиц и сохранение изображений
    for x, y, width, height in faces:
        cv2.rectangle(image, (x, y), (x + width, y + height), color=(0, 255, 0), thickness=3)
        cv2.imwrite("{}_result.jpg".format(number), image)
