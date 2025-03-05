# import sys
# from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout
# from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen
# from PyQt5.QtCore import Qt
#
# class ImageHighlighter(QWidget):
#     def __init__(self, original_image_path, binary_image_path):
#         super().__init__()
#         self.original_image_path = original_image_path
#         self.binary_image_path = binary_image_path
#         self.initUI()
#
#     def initUI(self):
#         # Layout
#         layout = QVBoxLayout()
#
#         # Original image label
#         self.original_image_label = QLabel(self)
#         pixmap = QPixmap(self.original_image_path)
#         self.original_image_label.setPixmap(pixmap)
#         layout.addWidget(self.original_image_label)
#
#         # Set the layout on the application's window
#         self.setLayout(layout)
#
#         self.setWindowTitle('Image Highlighter')
#         self.setGeometry(100, 100, pixmap.width(), pixmap.height())
#         self.show()
#
#     def paintEvent(self, event):
#         painter = QPainter(self)
#
#         # Draw original image
#         pixmap = QPixmap(self.original_image_path)
#         painter.drawPixmap(self.rect(), pixmap)
#
#         # Load binary image
#         binary_image = QImage(self.binary_image_path)
#
#         # Set the pen for highlighting
#         pen = QPen(Qt.red)
#         pen.setWidth(3)
#         painter.setPen(pen)
#
#         # Iterate over the binary image
#         for x in range(binary_image.width()):
#             for y in range(binary_image.height()):
#                 if binary_image.pixelColor(x, y) == Qt.white:  # White pixels represent features to highlight
#                     # Highlight with a red rectangle for now
#                     painter.drawEllipse(x, y, 1, 1)
#
#         painter.end()
#
# if __name__ == '__main__':
#     app = QApplication(sys.argv)
#     ex = ImageHighlighter('/home/weihuazhou/Desktop/automated-labelme/labelme/data/training/23613718_13.png', '/home/weihuazhou/Desktop/automated-labelme/labelme/data/predicted/23613718_13/13_p.png')
#     sys.exit(app.exec_())
import cv2
import numpy as np

# Load the original image and the binary image
original_image = cv2.imread('/home/weihuazhou/Desktop/automated-labelme/labelme/data/training/19995466_24.png')  # Replace with your original image file
binary_image = cv2.imread('/home/weihuazhou/Desktop/automated-labelme/labelme/data/predicted/19995466_24/24_p.png', cv2.IMREAD_GRAYSCALE)  # Replace with your binary image file
# Create a mask for highlighted features
highlight_mask = np.zeros_like(original_image)
highlight_mask[binary_image > 0] = (0, 0, 255)  # Red color in BGR format

# Create a final image with transparent highlights
transparency = 0.5  # Adjust the transparency as needed (0.0 for fully transparent, 1.0 for fully opaque)
final_image = cv2.addWeighted(original_image, 1, highlight_mask, transparency, 0)

# Display or save the final image
cv2.imshow('Highlighted Image', final_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
# If you want to save the result:
# cv2.imwrite('highlighted_image.jpg', final_image)


# If you want to save the result:
# cv2.imwrite('highlighted_image.jpg', final_image)
