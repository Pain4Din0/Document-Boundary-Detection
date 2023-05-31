from pyimagesearch.transform import four_point_transform

import numpy as np
import argparse
import cv2
import imutils

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True,
	help = "Path to the image to be scanned")
args = vars(ap.parse_args())

# 加载图像，计算图像比例
image = cv2.imread(args["image"])

# 调整图像尺寸
ratio = image.shape[0] / 500.0
orig = image.copy()
image = imutils.resize(image, height = 500)

# 灰阶处理
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 高斯模糊
gray = cv2.GaussianBlur(gray, (5, 5), 0)
# Canny
edged = cv2.Canny(gray, 75, 200)

print("STEP 1: Edge Detection")
cv2.imshow("Original", image)
cv2.imshow("Edged", edged)

# 寻找边缘图像中的轮廓，只保留最大的轮廓
cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# OpenCV 版本兼容性处理
cnts = cnts[1] if imutils.is_cv3() else cnts[0]

# 只取面积最大的5个轮廓
cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]

for c in cnts:
	# 计算轮廓线周长或曲线长度
	peri = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 0.01 * peri, True)
	# 当找到四个顶角时
	screenCnt = approx
	if len(approx) == 4:
		screenCnt = approx
		break
	
# 展示
print("STEP 2: Finding Boundary")

cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
cv2.imshow("Boundary", image)

cv2.waitKey(0)
cv2.destroyAllWindows()