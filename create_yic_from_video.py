import cv2
import numpy as np
import os


def create_images_for_person(person, video_dir):
    """Функция которая нарежет черно белые кадры из видое
    person - имя_фамилия человека"""
    # Поддержтваемые форматы видео:
    # 3gp, avi, f4v, hevc, mkv, mov, mp4, mpg, ts, webm, wmv

    SAVING_FRAMES_PER_SECOND = 17  # Интервал выборки кадров 17 если видео минута
    SAVING_PATH = f'images/{person}'  # Папка сохранения кадров
    # Если тдериктории не существует создадим ее
    if not os.path.exists(SAVING_PATH):
        os.mkdir(SAVING_PATH)
    cap = cv2.VideoCapture(video_dir)  # Открываем видео на обработку
    fps = cap.get(cv2.CAP_PROP_FPS)  # FPS исходного видео
    saving_frames_per_second = min(fps, SAVING_FRAMES_PER_SECOND)  # Выбираем меньшее значение FPS:

    # вдруг FPS исходного видео меньше чем задано для выборки
    frames_timecodes = []  # Получаем таймкоды на нужные кадры
    clip_duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)  # Длительность клипа
    for i in np.arange(0, clip_duration, 1 / saving_frames_per_second):
        frames_timecodes.append(i)
    cur_frame = 0  # Счетчик для всех кадров
    img_count = 1  # Счетчик для кадров, которые впоследствие будут отобраны
    while img_count <= 1000:  # Запускаем цикл выборки кадров
        is_read, frame = cap.read()  # Считываем кадр
        if not is_read:  # Если кадров больше не осталось, выходим из цикла
            break
        frame_duration = cur_frame / fps  # Вычисляем таймкод текущего кадра
        try:
            closest_duration = frames_timecodes[img_count - 1]  # Берем  таймкод из массива нужных таймкодов
        except IndexError:
            break  # Если массив закончился, то срабатывает исключение, т.е. все нужные кадры записаны
        # Если таймкод текущего кадра больше или равен чем таймкод из массива, значит сохраняем кадр
        if frame_duration >= closest_duration:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Переводим изображение в черно-белую палитру
            cv2.imwrite(os.path.join(SAVING_PATH, f"{img_count}.jpg"), gray)  # Записываем кадр в папку 'frames'
            img_count += 1  # Увеличиваем счетчик сохраненных кадров
        cur_frame += 1  # Увеличиваем счетчик всех пройденных кадров


import cv2
import os
import numpy as np
from PIL import Image

# Для детектирования лиц используем каскады Хаара
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

# Для распознавания используем локальные бинарные шаблоны
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('faces.yml')


def get_images(path):
    # Ищем все фотографии и записываем их в image_paths
    image_paths = [os.path.join(path, f) for f in os.listdir(path)]

    images = []
    labels = []

    for image_path in image_paths:
        # Переводим изображение в черно-белый формат и приводим его к формату массива
        gray = Image.open(image_path).convert('L')
        image = np.array(gray, 'uint8')
        # Из каждого имени файла извлекаем номер человека, изображенного на фото
        person = 2

        # Определяем области где есть лица
        faces = faceCascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40))
        # Если лицо нашлось добавляем его в список images, а соответствующий ему номер в список labels
        for (x, y, w, h) in faces:
            images.append(image[y: y + h, x: x + w])
            labels.append(person)
            cv2.imshow("", image[y: y + h, x: x + w])
            cv2.waitKey(50)
    return images, labels


path = 'photos'
# Получаем лица и соответствующие им номера
images, labels = get_images(path)
cv2.destroyAllWindows()

# Обучаем программу распознавать лица
recognizer.train(images, np.array(labels))
recognizer.save('faces.yml')

image_paths = [os.path.join(path, f) for f in os.listdir(path)]

# for image_path in image_paths:
#     # Ищем лица на фотографиях
#     gray = Image.open(image_path).convert('L')
#     image = np.array(gray, 'uint8')
#     faces = faceCascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
#
#     for (x, y, w, h) in faces:
#         # Если лица найдены, пытаемся распознать их
#         # Функция  recognizer.predict в случае успешного распознавания возвращает номер и параметр confidence,
#         # этот параметр указывает на уверенность алгоритма, что это именно тот человек, чем он меньше, тем больше уверенность
#         number_predicted, conf = recognizer.predict(image[y: y + h, x: x + w])
#
#         # Извлекаем настоящий номер человека на фото и сравниваем с тем, что выдал алгоритм
#         number_actual = 1
#
#         if number_actual == number_predicted:
#             print("{} is Correctly Recognized with confidence {}".format(number_actual, conf))
#         else:
#             print("{} is Incorrect Recognized as {}".format(number_actual, number_predicted))
#         cv2.imshow("Recognizing Face", image[y: y + h, x: x + w])
#         cv2.waitKey(1000)
if __name__ == '__main__':
    # Запуск программы первая имя фамилия вторая dir видео
    create_images_for_person("Kizhinov_Andrey", r"D:\My_Downloads\20211202_192435.mp4")
