import cv2
import os
import numpy as np
import csv
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import time
from datetime import datetime

# Путь к папке с датасетом лиц
dataset_dir = 'faces_dataset'
# Путь к файлу с контактами (CSV)
contacts_file = 'contacts.csv'
# Путь к файлу для записи данных о распознанных лицах
detected_faces_file = 'detected_faces_log.csv'

# Настройки SMTP для отправки письма
SMTP_SERVER = 'smtp.gmail.com'  # Например, для Gmail
SMTP_PORT = 587
SENDER_EMAIL = 'example@gmail.com'  # Ваш email
SENDER_PASSWORD = '123ali'  # Пароль или токен приложения
RECIPIENT_EMAIL = 'example@gmail.com'  # Email получателя

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


# Загрузка контактов из CSV-файла
def load_contacts(file_path):
    if not os.path.exists(file_path):
        print(f"Файл {file_path} не найден!")
        return {}
    contacts_dict = {}
    with open(file_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            name = row['Name']
            contact = row['Contact']
            contacts_dict[name] = contact
    return contacts_dict


contacts = load_contacts(contacts_file)


# Функция для записи данных о распознанных лицах в CSV-файл
def log_detected_face(name, info):
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    # Создаем или открываем файл для записи
    if not os.path.exists(detected_faces_file):
        with open(detected_faces_file, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['Name', 'Timestamp'])  # Заголовок столбцов
    # Добавляем новую запись
    with open(detected_faces_file, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([name, info, current_time])
    print(f"Записано: {name}, Время: {current_time}")


# Функция для отправки письма
def send_email(name, confidence, contact):
    try:
        msg = MIMEMultipart()
        msg['From'] = SENDER_EMAIL
        msg['To'] = RECIPIENT_EMAIL
        msg['Subject'] = f"Уведомление о распознавании: {name}"

        body = f"""
        Лицо было зафиксировано:
        Имя: {name}
        Уверенность: {confidence:.2f}%
        Контакт: {contact}
        """
        msg.attach(MIMEText(body, 'plain'))

        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()  # Защищенное соединение
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        text = msg.as_string()
        server.sendmail(SENDER_EMAIL, RECIPIENT_EMAIL, text)
        server.quit()
        print(f"Уведомление отправлено на {RECIPIENT_EMAIL}")
    except Exception as e:
        print(f"Ошибка отправки письма: {e}")


# Видеозахват с веб-камеры
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Ошибка: камера не доступна!")
    exit()

# Переменные для ограничения записей
last_record_time = time.time()
record_count = 0
hour_start_time = time.time()

while True:
    # Считываем кадр
    ret, frame = cap.read()
    if not ret:
        break
    # Преобразуем кадр в градации серого
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Детектируем лица
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(120, 120))
    # Для каждого обнаруженного лица
    for (x, y, w, h) in faces:
        face_roi = gray[y:y + h, x:x + w]  # Регион интереса (лицо)
        # Распознавание лица
        label, confidence = recognizer.predict(face_roi)
        if confidence >= 20:  # Можете настроить порог уверенности
            name = label_map.get(label, "Unknown")
            contact = contacts.get(name, "Нет информации")
            info_text = f"{name}, {confidence:.2f}"

            # Проверяем условия для записи
            current_time = time.time()
            elapsed_since_last_record = current_time - last_record_time
            elapsed_since_hour_start = current_time - hour_start_time

            if record_count < 10: # and elapsed_since_last_record >= 360:  # Записываем каждые 6 минут (360 секунд)
                log_detected_face(name, contact)
                last_record_time = current_time
                record_count += 1
            elif elapsed_since_hour_start >= 3600:  # Сброс счетчика каждые 60 минут
                record_count = 0
                hour_start_time = current_time

        else:
            name = "Unknown"
            info_text = f"Неизвестное лицо, Уверенность: {confidence:.2f}"

        # Рисуем прямоугольник вокруг лица
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, info_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # Отображаем кадр
    cv2.imshow('Face Recognition', frame)
    # Выход по нажатию клавиши ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Освобождаем ресурсы
cap.release()
cv2.destroyAllWindows()