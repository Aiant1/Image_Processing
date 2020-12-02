import numpy as np
import cv2 as cv


def dense_optical_flow():
    cap = cv.VideoCapture('sequence_gray.avi')
    ret, frame1 = cap.read()
    prvs = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    
    
    recap = cv.VideoWriter('dense_output_sequence_gray.mp4', cv.VideoWriter_fourcc(*'mp4v'), 4, (width,height))
    
    while(True):
        ret, frame2 = cap.read()
        if not ret:
            break
        else:
            next = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
            flow = cv.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 7, 1.5, 0)
            hsv = frame2.copy()
            h=hsv.shape[0]
            w=hsv.shape[1]
            for i in range(0,h,10):
                for j in range(0,w,10):
                    iflow, jflow = flow[i, j]
                    cv.line(hsv, (j, i), (int(j + iflow), int(i + jflow)), (0, 150, 240), 1)
            k = cv.waitKey(30) & 0xff
            if k == 27:
                break
            recap.write(hsv)
            prvs = next
    
    cap.release()
    recap.release()
    
    cv.destroyAllWindows()
    
if __name__ == '__main__':
    
    dense_optical_flow()
   
    cv.destroyAllWindows()