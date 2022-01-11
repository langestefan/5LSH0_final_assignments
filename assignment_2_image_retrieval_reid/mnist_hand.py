import cv2

# select the digit we want
digit = 9

# define path where the digit is stored
path = "handwritten/" + str(digit) + ".png"
print(path)

# read image, grayscale and change resolution to 28x28
img_gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
img_28 = cv2.resize(255 - img_gray, (28, 28), interpolation=cv2.INTER_LINEAR)

# contrast and brightness values
contrast = 5
brightness = -220

# improve brightness and contrast
img_28 = cv2.addWeighted(img_28, contrast, img_28, 0, brightness)

# show enhanced image
cv2.imshow("Resized image", img_28)
cv2.waitKey(0)
cv2.destroyAllWindows()

# write image to folder
print('Resized Dimensions : ', img_28.shape)
cv2.imwrite("processed/" + str(digit) + ".png", img_28)

