import numpy as np
from PIL import ImageGrab
import cv2
import time

def screen_record(): 
    probes = [cv2.imread("units/probe1.png",0),cv2.imread("units/probe2.png",0),cv2.imread("units/probe1.png",0)]
    probes_wh = [ i.shape[::-1] for i in probes]
    print(probes_wh)
    last_time = time.time()
    while(True):
        # 800x600 windowed mode
        printscreen =  np.array(ImageGrab.grab(bbox=(8,40,1032,744)))
        print('loop took {} seconds'.format(time.time()-last_time))
        last_time = time.time()
        gray_image = cv2.cvtColor(printscreen, cv2.COLOR_BGR2GRAY)
        for n,i in enumerate(probes):
            res = cv2.matchTemplate(gray_image,i,cv2.TM_CCOEFF_NORMED)
            threshold = 0.6
            loc = np.where( res >= threshold)
            for pt in zip(*loc[::-1]):
                cv2.rectangle(printscreen, pt, (pt[0] + probes_wh[n][0], pt[1] +  probes_wh[n][1]), (0,255,255), 2)
            loc = None

        cv2.imshow('window',cv2.cvtColor(printscreen, cv2.COLOR_BGR2RGB))
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

screen_record()
