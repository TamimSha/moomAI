from scripts import video_renderer, frame_eraser
from data import data

done = video_renderer()
if(done):
    done = frame_eraser()

print(done)