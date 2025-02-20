import cv2
import os

# Создание папки для нового пользователя
user_name = input("Введите имя пользователя: ")
dataset_dir = 'faces_dataset'
user_dir = os.path.join(dataset_dir, user_name)
os.makedirs(user_dir, exist_ok=True)

# Инициализация камеры
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face_roi = frame[y:y + h, x:x + w]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Сохранение изображения лица
        save_path = os.path.join(user_dir, f"{count}.jpg")
        cv2.imwrite(save_path, face_roi)
        print(f"Сохранено: {save_path}")
        count += 1

    cv2.imshow('Collecting Faces', frame)

    if cv2.waitKey(1) & 0xFF == 27 or count >= 50:  # Нажмите ESC или соберите 50 изображений
        break

cap.release()
cv2.destroyAllWindows()