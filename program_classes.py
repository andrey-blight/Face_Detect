import sys
import cv2
import pickle
from PyQt5 import uic
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *


class FilterDial(QDialog):
    def __init__(self):
        super(FilterDial, self).__init__()
        uic.loadUi('design/filters.ui', self)
        self.accept_btn.setIcon(QIcon('photos/check.png'))
        self.accept_btn.clicked.connect(self.accept)
        self.slider_blur.valueChanged.connect(self.lcdNumber_blur.display)
        self.slider_sharpen.valueChanged.connect(self.lcdNumber_sharpen.display)
        self.blur.stateChanged.connect(self.onStateChange)
        self.monochrome.stateChanged.connect(self.onStateChange)
        self.sharpen.stateChanged.connect(self.onStateChange)

    def onStateChange(self):
        boxes = [self.blur, self.monochrome, self.sharpen]
        uncheckeds = [box for box in boxes if not box.isChecked()]
        for box in uncheckeds:
            box.setDisabled(len(uncheckeds) == 2)

    def get_data(self):
        filters = [el for el in [self.blur, self.monochrome, self.sharpen] if el.isChecked()]
        if filters:
            if filters[0] == self.sharpen:
                return filters[0].text().strip(), self.slider_sharpen.value() / 10
            if filters[0] == self.blur:
                return filters[0].text().strip(), self.slider_blur.value() / 10
            if filters[0] == self.monochrome:
                return filters[0].text().strip(), None
        return None, None


class BD(QMainWindow):
    def __init__(self):
        super(BD, self).__init__()
        uic.loadUi('design/bd_people.ui', self)
        self.delete_btn.setIcon(QIcon('photos/garbige.png'))
        self.download_btn.setIcon(QIcon('photos/video.png'))
        self.tableWidget.setColumnWidth(0, 150)
        self.tableWidget.setColumnWidth(1, 300)


class CameraCv(QThread):
    changePixmap = pyqtSignal(QImage)

    def __init__(self, parent):
        super(CameraCv, self).__init__()
        self.stop = False  # Флаг остановки видеопотока
        self.blur = False  # Флаг размытия изображения
        self.degree_blur = None  # Степень размытия
        self.monochrome = False  # Флаг чб
        self.sharpen = False  # Флаг контраста изображения
        self.degree_sharpen = None  # Степень контраста

    def face_id(self, recognizer, labels, grey_frame, color_frame, x, y, w, h):
        roi_gray = grey_frame[y:y + h, x:x + w]
        id_, conf = recognizer.predict(roi_gray)
        print(conf)
        if conf <= 70:
            print(labels[id_])
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color = (255, 255, 255)
            stroke = 2
            cv2.putText(color_frame, name, (x, y), font, 1, color, stroke, cv2.LINE_AA)
        color = (255, 0, 0)
        stroke = 2
        end_cord_x = x + w
        end_cord_y = y + h
        cv2.rectangle(color_frame, (x, y), (end_cord_x, end_cord_y), color, stroke)
        return color_frame

    def do_sharpen(self, frame):
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)  # Converting image to LAB Color model
        l, a, b = cv2.split(lab)  # Splitting the LAB image to different channels
        clahe = cv2.createCLAHE(clipLimit=max(1, self.degree_sharpen),
                                tileGridSize=(7, 7))  # Applying CLAHE to L-channel
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))  # Merge the CLAHE enhanced L-channel with the a and b channel
        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)  # Converting image from LAB Color model to RGB model

    def do_blur(self, frame):
        print(self.degree_blur)
        return cv2.GaussianBlur(frame, (7, 7), max(1, self.degree_blur))

    def do_monochrome(self, frame):
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def run(self):
        """Здесь будет проискходить работа с камерой
        (получение видое, распознавание лиц, накладывание фильтров)"""
        self.stop = False
        # TODO Возможно сделать привязку к self чтобы не передавать
        face_cascade = cv2.CascadeClassifier('data/head_cascade.xml')
        # eye_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_eye.xml')
        # smile_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_smile.xml')
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read(r"data/recognition.yml")
        with open("data/labels.pickle", 'rb') as f:
            og_labels = pickle.load(f)
            labels = {v: k for k, v in og_labels.items()}
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            # TODO Возможность добавить красный экран вместо последнего кадра видое
            if ret and not self.stop:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=8, minSize=(60, 60))
                for (x, y, w, h) in faces:
                    frame = self.face_id(recognizer, labels, gray, frame, x, y, w, h)
                if self.sharpen:
                    frame = self.do_sharpen(frame)
                elif self.blur:
                    frame = self.do_blur(frame)
                elif self.monochrome:
                    frame = self.do_monochrome(frame)
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                convert_qt = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                p = convert_qt.scaled(640, 480, Qt.KeepAspectRatio)  # Здесь задоются размеры окна видео
                self.changePixmap.emit(p)

    def stop_show(self):
        self.stop = True  # Выключаем видео

    def set_data(self, name, coeff):
        self.stop = self.blur = self.monochrome = self.sharpen = False
        self.degree_blur = self.degree_sharpen = None
        if name == "sharpen":
            self.sharpen = True
            self.degree_sharpen = coeff
        elif name == "blur":
            self.blur = True
            self.degree_blur = coeff
        elif name == "monochrome":
            self.monochrome = True


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        uic.loadUi('design/camera.ui', self)
        # TODO отправить этот код в файл дизайна
        self.camera.setIcon(QIcon('photos/pngwing.com.png'))
        self.filters.setIcon(QIcon('photos/filters.png'))
        self.bd_of_people.setIcon(QIcon('photos/audience (1).png'))
        # Создание подключений
        self.filters.clicked.connect(self.open_filter)
        self.bd_of_people.clicked.connect(self.open_bd)
        # Подключение видеопотока
        self.camera = CameraCv(self)
        self.camera.changePixmap.connect(self.setImage)
        self.camera.start()

    def setImage(self, image):
        self.opencv_label.setPixmap(QPixmap.fromImage(image))

    def open_filter(self):
        dlg_filter = FilterDial()
        if dlg_filter.exec():
            self.camera.set_data(*dlg_filter.get_data())

    def open_bd(self):
        self.dialog_bd = BD()
        self.dialog_bd.show()


def except_hook(cls, exception, traceback):
    sys.__excepthook__(cls, exception, traceback)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MainWindow()
    ex.show()
    sys.excepthook = except_hook
    sys.exit(app.exec())
