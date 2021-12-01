import cv2 as cv

BLACK_WIGHT = False  # Фильтр чб
BLUR = False  # Фильтр блура
CONTRAST = False  # Фильтр контраста
cap = cv.VideoCapture(0)  # Получение видеоряда с первой камеры
cascade = cv.CascadeClassifier("lbpcascade_frontalface.xml")  # Загрузка каскада
while True:
    ok, img = cap.read()  # Получение картинки
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # Создание картинки серого цвета
    sf = min(640. / img.shape[1], 480. / img.shape[0])  # Получение размеров для сжатия
    gray = cv.resize(gray, (0, 0), None, sf, sf)  # Сжатие серой картинки
    gray = cv.GaussianBlur(gray, (5, 5), 1.5)  # Блур серой картинки для каскада
    rects = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                     minSize=(100, 100))  # Получение координат от каскада
    if CONTRAST:
        degree = 1.5  # Степень контраста
        lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)  # Converting image to LAB Color model
        l, a, b = cv.split(lab)  # Splitting the LAB image to different channels
        clahe = cv.createCLAHE(clipLimit=degree, tileGridSize=(8, 8))  # Applying CLAHE to L-channel
        cl = clahe.apply(l)
        limg = cv.merge((cl, a, b))  # Merge the CLAHE enhanced L-channel with the a and b channel
        img = cv.cvtColor(limg, cv.COLOR_LAB2BGR)  # Converting image from LAB Color model to RGB model
    if BLUR:
        degree = 2  # Степень блура (предлагаю через фронтенд возможновсть выбора от 1(нет блура) до 10(сильный блур))
        img = cv.GaussianBlur(img, (7, 7), 10)  # Блур изображение
    if BLACK_WIGHT:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # Черно белая картинка
    for x, y, w, h in rects:
        cv.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Рисование прямоугольников
    cv.imshow("Case", img)  # Показ видеоряда
    if cv.waitKey(3) > 0:  # Выход из приложениея
        break
