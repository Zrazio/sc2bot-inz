import numpy as np
from PIL import ImageGrab
import cv2
import time

def screen_record(): 
    probes = [cv2.imread("units/probe1.png",0),cv2.imread("units/probe2.png",0),cv2.imread("units/probe1.png",0)]
    probes_wh = [ i.shape[::-1] for i in probes]
    print(probes_wh)

    #KAZEKAZEKAZEKAZE
    #cv2.imshow('probe', probes[0])
    #for i in probes:

    #    sift = cv2.KAZE_create()
    #    (kps, descs) = sift.detectAndCompute(i, None)
    #    print("# kps: {}, descriptors: {}".format(len(kps), descs.shape))
    last_time = time.time()
    while(True):
        # 800x600 windowed mode
        printscreen =  np.array(ImageGrab.grab(bbox=(8,40,1032,744)))
        print('loop took {} seconds'.format(time.time()-last_time))
        last_time = time.time()
        gray_image = cv2.cvtColor(printscreen, cv2.COLOR_BGR2GRAY)
        #KAZEKAZEKAZEKAZE
        #ret = sift.detect(printscreen)
        #print(ret)
        for probe,probe_wh in zip(probes,probes_wh):
            r = cv2.matchTemplate(probe,gray_image, cv2.TM_CCORR_NORMED)
            print(r)
            threshold = 0.87
            loc = np.where( r >= threshold)
            for pt in zip(*loc[::-1]):
                cv2.rectangle(printscreen,
                              pt,(pt[0]+probe_wh[0],pt[1]+probe_wh[1]),(0,0,255),2)
            
        cv2.imshow('window',cv2.cvtColor(printscreen, cv2.COLOR_BGR2RGB))
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

screen_record()
