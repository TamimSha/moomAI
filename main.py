from scripts import video_renderer, frame_eraser, diversifier
from data import data

#diversifier()

done = video_renderer()
if(done):
    done = frame_eraser()
if(done):
    done = diversifier()
print(done)
