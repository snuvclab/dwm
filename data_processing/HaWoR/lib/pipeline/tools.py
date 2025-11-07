import cv2
from tqdm import tqdm
import numpy as np
import torch

from ultralytics import YOLO


if torch.cuda.is_available():
    autocast = torch.cuda.amp.autocast
else:
    class autocast:
        def __init__(self, enabled=True):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass


def detect_track(imgfiles, thresh=0.5):
    
    hand_det_model = YOLO('./weights/external/detector.pt')

    # Run
    boxes_ = []
    tracks = {}
    for t, imgpath in enumerate(tqdm(imgfiles)):
        img_cv2 = cv2.imread(imgpath)

        ### --- Detection ---
        with torch.no_grad():
            with autocast():
                results = hand_det_model.track(img_cv2, conf=thresh, persist=True, verbose=False)
                
                boxes = results[0].boxes.xyxy.cpu().numpy()
                confs = results[0].boxes.conf.cpu().numpy()
                handedness = results[0].boxes.cls.cpu().numpy()
                if not results[0].boxes.id is None:
                    track_id = results[0].boxes.id.cpu().numpy()
                else:
                    track_id = [-1] * len(boxes)

                boxes = np.hstack([boxes, confs[:, None]])
                find_right = False
                find_left = False
                for idx, box in enumerate(boxes):
                    if track_id[idx] == -1:
                        if handedness[[idx]] > 0:
                            id = int(10000)
                        else:
                            id = int(5000)
                    else:
                        id = track_id[idx]
                    subj = dict()
                    subj['frame'] = t 
                    subj['det'] = True
                    subj['det_box'] = boxes[[idx]]
                    subj['det_handedness'] = handedness[[idx]]
                    
                    
                    if (not find_right and handedness[[idx]] > 0) or (not find_left and handedness[[idx]]==0):
                        if id in tracks:
                            tracks[id].append(subj)
                        else:
                            tracks[id] = [subj]

                        if handedness[[idx]] > 0:
                            find_right = True
                        elif handedness[[idx]] == 0:
                            find_left = True
    tracks = np.array(tracks, dtype=object)
    boxes_ = np.array(boxes_, dtype=object)

    return boxes_, tracks


def parse_chunks(frame, boxes, min_len=16):
    """ If a track disappear in the middle, 
     we separate it to different segments to estimate the HPS independently. 
     If a segment is less than 16 frames, we get rid of it for now. 
     """
    frame_chunks = []
    boxes_chunks = []
    step = frame[1:] - frame[:-1]
    step = np.concatenate([[0], step])
    breaks = np.where(step != 1)[0]

    start = 0
    for bk in breaks:
        f_chunk = frame[start:bk]
        b_chunk = boxes[start:bk]
        start = bk
        if len(f_chunk)>=min_len:
            frame_chunks.append(f_chunk)
            boxes_chunks.append(b_chunk)

        if bk==breaks[-1]:  # last chunk
            f_chunk = frame[bk:]
            b_chunk = boxes[bk:]
            if len(f_chunk)>=min_len:
                frame_chunks.append(f_chunk)
                boxes_chunks.append(b_chunk)

    return frame_chunks, boxes_chunks

def parse_chunks_hand_frame(frame):
    """ If a track disappear in the middle, 
     we separate it to different segments to estimate the HPS independently. 
     If a segment is less than 16 frames, we get rid of it for now. 
     """
    frame_chunks = []
    step = frame[1:] - frame[:-1]
    step = np.concatenate([[0], step])
    breaks = np.where(step != 1)[0]

    start = 0
    for bk in breaks:
        f_chunk = frame[start:bk]
        start = bk
        if len(f_chunk) > 0:
            frame_chunks.append(f_chunk)

        if bk==breaks[-1]:  # last chunk
            f_chunk = frame[bk:]
            if len(f_chunk) > 0:
                frame_chunks.append(f_chunk)

    return frame_chunks
