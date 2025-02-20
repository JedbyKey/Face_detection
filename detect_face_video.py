import cv2
import os
import numpy as np

# Путь к папке с датасетом лиц
dataset_dir = 'faces_dataset'

# Инициализация распознавателя лиц
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Загрузка каскада для детекции лиц
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Функция для загрузки датасета
def load_dataset():
    images = []
    labels = []
    label_id = 0
    label_map = {}
    # Проходим по всем папкам в dataset_dir
    for subdir in os.listdir(dataset_dir):
        subpath = os.path.join(dataset_dir, subdir)
        if os.path.isdir(subpath):
            label_map[label_id] = subdir  # Сохраняем соответствие ID и имени
            for filename in os.listdir(subpath):
                img_path = os.path.join(subpath, filename)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Загружаем изображение в градациях серого
                if img is not None:
                    images.append(np.array(img, dtype='uint8'))
                    labels.append(label_id)
            label_id += 1
    return images, np.array(labels), label_map

# Загрузка датасета
images, labels, label_map = load_dataset()
print(f"Загружено {len(images)} изображений и {len(labels)} меток.")

# Обучение модели
recognizer.train(images, labels)
print("Модель обучена!")

# Видеозахват с веб-камеры
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Ошибка: камера не доступна!")
    exit()

while True:
    # Считываем кадр
    ret, frame = cap.read()
    if not ret:
        break

    # Преобразуем кадр в градации серого
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Детектируем лица
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Для каждого обнаруженного лица
    for (x, y, w, h) in faces:
        face_roi = gray[y:y + h, x:x + w]  # Регион интереса (лицо)
        # Распознавание лица
        label, confidence = recognizer.predict(face_roi)
        if confidence < 50:  # Можете настроить порог уверенности
            name = label_map.get(label, "Unknown")
            print(f"Распознано: {name}, Уверенность: {confidence:.2f}")
        else:
            name = "Unknown"
        # Рисуем прямоугольник вокруг лица
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, f"{name} ({confidence:.2f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Отображаем кадр
    cv2.imshow('Face Recognition', frame)

    # Выход по нажатию клавиши ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Освобождаем ресурсы
cap.release()
cv2.destroyAllWindows()