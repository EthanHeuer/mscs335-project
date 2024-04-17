from GenLib import CreateNumbers
import cv2 as cv
import numpy as np

cn = CreateNumbers()
current_number = 0
current_std_scale = 1.0
canvas = np.zeros((10 * 28, 10 * 28))

window_name = "Generated Numbers"
trackbar_name = "Std %"


def updateTrackbar(val):
    global current_std_scale, current_number
    current_std_scale = val/100.0
    numbers = cn.getNumbers(number=current_number, count=100, std_scale=current_std_scale)
    for row in range(10):
        for col in range(10):
            canvas[row * 28:(row + 1) * 28, col * 28:(col + 1) * 28] = numbers[row * 10 + col]
    cv.imshow(window_name, np.repeat(np.repeat(canvas, 3, axis=0), 3, axis=1))


cv.namedWindow(window_name)
updateTrackbar(100)
cv.createTrackbar(trackbar_name, window_name, 100, 200, updateTrackbar)

while(True):
    k = cv.waitKey(1)
    if ord('0') <= k <= ord('9'):
        current_number = k-ord('0')
        updateTrackbar(cv.getTrackbarPos(trackbar_name, window_name))
    elif k != -1:
        break

cv.destroyAllWindows()
