from PyQt5.QtWidgets import QMainWindow

def get_3x3_neighbors(x, y):
    neighbors = [(x - 1, y - 1),
                 (x, y - 1),
                 (x + 1, y - 1),
                 (x - 1, y),
                 (x, y),
                 (x + 1, y),
                 (x - 1, y + 1),
                 (x, y + 1),
                 (x + 1, y + 1)]
    return neighbors

def get_neighbor_values(im, list_neighbors):
    list_value = []
    for point in list_neighbors:
        list_value.append(im[point[0], point[1]])

    return list_value

def get_matrix_33_filtering(matrix1, matrix2):
    import numpy as np
    matrix1, matrix2 = np.asarray(matrix1), np.asarray(matrix2)

    total = 0
    for i in range(3):
        for j in range(3):
            total += matrix1[i, j] * matrix2[i, j]
    return total/9

class image_processing_class(QMainWindow):
    def __init__(self):
        from PyQt5.uic import loadUi

        super(image_processing_class, self).__init__()
        loadUi('image_processing_tool.ui', self)
        self.source_image = None
        self.btnOpenimage.clicked.connect(lambda: self.open_image())
        self.btnShowhist.clicked.connect(lambda: self.show_histogram())
        self.btnConvert.clicked.connect(lambda: self.convert_image())
        self.btnEnhanceimage.clicked.connect(lambda: self.hist_equalization())
        self.btnBlur.clicked.connect(lambda: self.blur_image())
        self.btnSmooth.clicked.connect(lambda: self.smooth_image())
        self.btnSharpen.clicked.connect(lambda: self.sharpe_image())
        self.btnRotate.clicked.connect(lambda: self.rotate_image())
        self.btnScale.clicked.connect(lambda: self.scale_image())
        self.btnFlipLR.clicked.connect(lambda: self.flipLR_image())
        self.btnFlipTB.clicked.connect(lambda: self.flipTB_image())
        self.btnWarpCols.clicked.connect(lambda: self.warpCols_image())
        self.btnWarpRows.clicked.connect(lambda: self.warpRows_image())
        self.btnCrop.clicked.connect(lambda: self.crop_image())
        self.btnConnectWebcam.clicked.connect(lambda: self.connectwebcam())
        self.btnStopWebcam.clicked.connect(lambda: self.stopwebcam())
        self.btnConvertCam.clicked.connect(lambda: self.set_convert())
        self.btnRotateCam.clicked.connect(lambda: self.set_rotate())

        self.stop_webcam = False
        self.convert_flag = False
        self.rotate_flag = False

    def open_image(self):
        from PyQt5 import QtWidgets, QtCore
        from skimage import io
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, 'Open File', QtCore.QDir.rootPath(), '*.*')
        try:
            self.source_image = io.imread(fileName)
            self.show_image(self.lblImage1, self.source_image)
        except Exception as e:
            print('Error: {}'.format(e))

    def show_image(self, label, image):
        import qimage2ndarray
        from PyQt5 import QtGui

        image = qimage2ndarray.array2qimage(image)
        qpixmap = QtGui.QPixmap.fromImage(image)
        label.setPixmap(qpixmap)

    def show_histogram(self):
        from matplotlib import pyplot as plt
        from skimage import io

        plt.figure(figsize=(5, 4))
        plt.hist(self.source_image.ravel(), bins=256)
        plt.savefig('hist.png')
        hist_img = io.imread('hist.png')
        self.show_image(self.lblImage2, hist_img)

    def convert_image(self):
        from skimage.color import rgb2gray
        from skimage import io

        gray_image = rgb2gray(self.source_image)
        io.imsave('convert.png', gray_image)
        image2 = io.imread('convert.png')
        self.show_image(self.lblImage2, image2)

    def hist_equalization(self):
        from skimage import io, exposure

        im_equalization = exposure.equalize_hist(self.source_image)
        io.imsave('equalization.png', im_equalization)
        image2 = io.imread('equalization.png')
        self.show_image(self.lblImage2, image2)

    def blur_image(self):
        from skimage import filters, io

        try:
            blur = int(self.txtBlur.toPlainText())
            io.imsave('blurImage.png', filters.gaussian(self.source_image, sigma=blur, multichannel=True))
            im2 = io.imread('blurImage.png')
            self.show_image(self.lblImage2, im2)
        except Exception as e:
            print('Error: {}'.format(e))

    def smooth_image(self):
        from skimage import io
        from skimage.color import rgb2gray
        import numpy as np

        im = rgb2gray(self.source_image)
        im_gray_pad = np.pad(im, pad_width=1)

        n, m = im_gray_pad.shape[0], im_gray_pad.shape[1]
        im_filter = np.zeros(im_gray_pad.shape)

        weight = [[1, 0, -1],
                  [2, 0, -2],
                  [1, 0, -1]]

        for x in range(1, n - 1):
            for y in range(1, m - 1):
                neighbors = get_3x3_neighbors(x, y)
                neighbors_value = get_neighbor_values(im_gray_pad, neighbors)
                neighbors_33 = np.asarray(neighbors_value).reshape((3, 3))
                im_filter[x, y] = get_matrix_33_filtering(neighbors_33, weight)

        io.imsave('smoothing.png', im_filter)
        im2 = io.imread('smoothing.png')
        self.show_image(self.lblImage2, im2)

    def sharpe_image(self):
        import cv2
        import numpy as np
        from skimage import io

        image = cv2.cvtColor(self.source_image, cv2.COLOR_BGR2RGB)
        sharpening = np.array([[-1, -1, -1],
                               [-1, 9, -1],
                               [-1, -1, -1]])
        sharpening_out = cv2.filter2D(image, -1, sharpening)
        io.imsave('sharpenImage.png', sharpening_out)
        image2 = io.imread('sharpenImage.png')
        self.show_image(self.lblImage2, image2)

    def rotate_image(self):
        from skimage import io
        from skimage.transform import rotate

        alpha = int(self.txtRotate.toPlainText())
        im_rotated = rotate(self.source_image, int(alpha))
        io.imsave('rotateImage.png', im_rotated)
        image2 = io.imread('rotateImage.png')
        self.show_image(self.lblImage2, image2)

    def scale_image(self):
        from skimage import io

        im = io.imread(self.source_image)

        size0 = im.size[0], im.size[1]
        alpha = int(self.txtScaleCol.toPlainText())
        beta = int(self.txtScaleRow.toPlainText())
        size1 = (alpha, beta)
        im_scale = im.resize((round(size0[0] * size1[0]), round(size0[1] * size1[1])))

        self.show_image(self.lblImage2, im_scale)

    def flipLR_image(self):
        from PyQt5 import QtWidgets, QtCore
        from PIL import Image
        from skimage import io

        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, 'Open File', QtCore.QDir.rootPath(), '*.*')

        im = Image.open(fileName)
        im_flip_left_right = im.transpose(Image.FLIP_LEFT_RIGHT)
        io.imsave('scaleImage.png', im_flip_left_right)
        im2 = io.imread('scaleImage.png')
        self.show_image(self.lblImage2, im2)

    def flipTB_image(self):
        from PIL import Image
        from skimage import io

        im_flip_top_bottom = self.source_image.transpose(Image.FLIP_TOP_BOTTOM)
        io.imsave('flipTBImage.png', im_flip_top_bottom)
        image2 = io.imread('flipTBImage.png')
        self.show_image(self.lblImage2, image2)

    def warpCols_image(self):
        import numpy as np
        from PIL import Image
        from skimage import io
        import math

        img = Image.open(self.source_image).convert("L")
        img = np.array(img)
        rows, cols = img.shape[0], img.shape[1]
        img_output = np.zeros((rows, cols))

        for i in range(rows):
            for j in range(cols):
                offset_x = int(40.0 * math.sin(2 * 3.14 * i / 180))
                if j + offset_x < rows:
                    img_output[i, j] = img[i, (j + offset_x) % cols]
                else:
                    img_output[i, j] = 0

        io.imsave('warpColsImage.png', img_output)
        image2 = io.imread('warpColsImage.png')
        self.show_image(self.lblImage2, image2)

    def warpRows_image(self):
        import numpy as np
        from PIL import Image
        from skimage import io
        import math

        im = io.imread(self.source_image)
        img = Image.open(im).convert("L")
        img = np.array(img)
        rows, cols = img.shape[0], img.shape[1]
        img_output = np.zeros((rows, cols))

        for i in range(rows):
            for j in range(cols):
                offset_y = int(100.0 * math.sin(2 * 3.14 * j / 180))
                if i + offset_y < rows:
                    img_output[i, j] = img[(i + offset_y) % rows, j]
                else:
                    img_output[i, j] = 0

        io.imsave('warpRowsImage.png', img_output)
        image2 = io.imread('warpRowsImage.png')
        self.show_image(self.lblImage2, image2)

    def crop_image(self):

        x1 = int(self.txtCropX1.toPlainText())
        x2 = int(self.txtCropX2.toPlainText())
        y1 = int(self.txtCropY1.toPlainText())
        y2 = int(self.txtCropY2.toPlainText())

        crop_img = self.source_image[x1:x2, y1:y2]
        self.show_image(self.lblImage2, crop_img)

    def connectwebcam(self):
        import cv2

        self.stop_webcam = False
        self.convert_flag = False
        self.rotate_flag = False

        cap = cv2.VideoCapture(0)
        while True:
            ret, self.source_image = cap.read()
            self.source_image = cv2.cvtColor(self.source_image, cv2.COLOR_BGR2RGB)
            self.show_image(self.lblImage1, self.source_image)
            if self.convert_flag:
                convert_image = cv2.cvtColor(self.source_image, cv2.COLOR_BGR2GRAY)
                self.show_image(self.lblImage2, convert_image)
            if self.rotate_flag:
                rotate = int(self.txtRotateCam.toPlainText())
                if rotate == 90:
                    rotate_image = cv2.rotate(self.source_image, cv2.ROTATE_90_CLOCKWISE)
                elif rotate == 180:
                    rotate_image = cv2.rotate(self.source_image, cv2.ROTATE_180)
                elif rotate == 270:
                    rotate_image = cv2.rotate(self.source_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                else:
                    self.rotate_flag = False
                self.show_image(self.lblImage2, rotate_image)

            cv2.waitKey(24)
            if self.stop_webcam:
                break

        cap.release()
        cv2.destroyAllWindows()

    def stopwebcam(self):
        self.stop_webcam = True

    def set_convert(self):
        self.convert_flag = True

    def set_rotate(self):
        self.rotate_flag = True

def image_processing_tool():
    import sys
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    window = image_processing_class()
    window.show()
    app.exec()

if __name__ == "__main__":
    image_processing_tool()

# image_processing_tool()