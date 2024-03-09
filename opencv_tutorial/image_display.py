import cv2
import matplotlib.pyplot as plt 
from IPython.display import Image

cb_img = cv2.imread('checkerboard_color.png')
coke_img = cv2.imread('coca_cola_logo.png')

plt.imshow(cb_img)
plt.title('matplotlib imshow')
plt.show()

window1 = cv2.namedWindow('w1')
cv2.imshow(window1, cb_img)
cv2.waitKey(8000)
cv2.destroyWindow(window1)

window2 = cv2.namedWindow('w2')
cv2.imshow(window2, coke_img)
cv2.waitKey(8000) # wait 8s
cv2.destroyWindow(window2)

# wait until any key is pressed
window3 = cv2.namedWindow('w3')
cv2.imshow(window3, cb_img)
cv2.waitKey(0)
cv2.destroyWindow(window3)


window4 = cv2.namedWindow('w4')
alive = True
while alive:
    cv2.imshow(window4, coke_img)
    keypress = cv2.waitKey(1)
    if keypress == ord('q'): # wait until q is pressed
        alive = False

cv2.destroyWindow(window4)
cv2.destroyAllWindows()

stop = 1

Image(filename="checkerboard_18x18.png")

# retval = cv2.imread( filename[, flags] )
plt.imshow(cb_img, cmap="gray")

img_NZ_bgr = cv2.imread("New_Zealand_Lake.jpg", cv2.IMREAD_COLOR)
b, g, r = cv2.split(img_NZ_bgr)

# Show the channels
plt.figure(figsize=[20, 5])

plt.subplot(141);plt.imshow(r, cmap="gray");plt.title("Red Channel")
plt.subplot(142);plt.imshow(g, cmap="gray");plt.title("Green Channel")
plt.subplot(143);plt.imshow(b, cmap="gray");plt.title("Blue Channel")

# Merge the individual channels into a BGR image
imgMerged = cv2.merge((b, g, r))
# Show the merged output
plt.subplot(144)
plt.imshow(imgMerged[:, :, ::-1])
plt.title("Merged Output")

img_NZ_rgb = cv2.cvtColor(img_NZ_bgr, cv2.COLOR_BGR2RGB)
plt.imshow(img_NZ_rgb)

img_hsv = cv2.cvtColor(img_NZ_bgr, cv2.COLOR_BGR2HSV)

# Split the image into the B,G,R components
h,s,v = cv2.split(img_hsv)

# Show the channels
plt.figure(figsize=[20,5])
plt.subplot(141);plt.imshow(h, cmap="gray");plt.title("H Channel")
plt.subplot(142);plt.imshow(s, cmap="gray");plt.title("S Channel")
plt.subplot(143);plt.imshow(v, cmap="gray");plt.title("V Channel")
plt.subplot(144);plt.imshow(img_NZ_rgb);   plt.title("Original")

h_new = h + 10
img_NZ_merged = cv2.merge((h_new, s, v))
img_NZ_rgb = cv2.cvtColor(img_NZ_merged, cv2.COLOR_HSV2RGB)

cv2.imwrite("New_Zealand_Lake_SAVED.png", img_NZ_bgr)

Image(filename='New_Zealand_Lake_SAVED.png')

