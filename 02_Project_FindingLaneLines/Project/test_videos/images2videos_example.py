import cv2
import os

fps = 25
fourcc = cv2.VideoWriter_fourcc('I', '4', '2', '0')
video_writer = cv2.VideoWriter(filename='challenge/result.avi', fourcc=fourcc, fps=fps, frameSize=(1280, 720))
for i in range(0, 244):
    img = cv2.imread("challenge/%d.jpg" % i)
    cv2.waitKey(100)
    video_writer.write(img)
    print(str(i) + '/' + str(243))
video_writer.release()
