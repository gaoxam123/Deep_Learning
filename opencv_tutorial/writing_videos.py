import cv2
import matplotlib.pyplot as plt

source = './race_car.mp4' # source = 0 for webcam

cap = cv2.VideoCapture(source)

if(cap.isOpened()==False):
    print('error')

ret, frame = cap.read()
plt.imshow(frame[...,::-1]) # first frame of the video

# VideoWriter

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Define the codec and create VideoWriter object.
# VideoWriter object = cv.VideoWriter(filename, fourcc, fps, frameSize )
out_avi = cv2.VideoWriter("race_car_out.avi", cv2.VideoWriter_fourcc("M", "J", "P", "G"), 10, (frame_width, frame_height))

out_mp4 = cv2.VideoWriter("race_car_out.mp4", cv2.VideoWriter_fourcc(*"XVID"), 10, (frame_width, frame_height))

while(cap.isOpened()):
    ret, frame = cap.read()

    if ret == True:
        out_avi.write(frame)
        out_mp4.write(frame)
    
    else:
        break

cap.release()
out_avi.release()
out_mp4.release()