import progressbar
import time
import cv2
import sys
import threading
from FrameRenderer import FrameRenderer
import json

def video_renderer():

    try:
        with open('./data/video.json', 'r') as data:
            data = data.read()
        files = json.loads(data)
    except:
        return 0

    threads = []
    fileNum = files['count']

    for i in range(0, fileNum):
        frameRenderer = FrameRenderer(files['files'][i]['name'] + "_",
            files['output'],
            files['location']+files['files'][i]['name']+files['files'][i]['type'])
        threads.append(frameRenderer)
    
    for t in threads:
        if(t.open):
            t.start()

    try:
        with progressbar.ProgressBar(max_value=100.0) as bar:
            active = True
            while(active):
                isDead = True
                progress = 0.
                for t in threads:
                    isDead = isDead and not t.isAlive()
                    progress += t.getProgress()
                if(isDead):
                    active = False
                progress = min(round(100.0 * progress / fileNum, 2), 100)
                #progress = progress if (progress < 100) else 100.0
                bar.update(progress)
                time.sleep(0.1)
    except KeyboardInterrupt:
        for t in threads:
            t.kill()
            t.join()
        return 0

    for t in threads:
        t.join()

    return 1
