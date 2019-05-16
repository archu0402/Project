import sys
import numpy as np
import pandas as pd
if len(sys.argv) < 2:
    print("Usage: $ python {0} [video_path] [output_path(optional)]", sys.argv[0])
    exit()

from yolo import YOLO
from yolo import detect_video

if __name__ == '__main__':
    video_path = sys.argv[1]

    new_obj=YOLO()
    try:
        if len(sys.argv) > 2:
            output_path = sys.argv[2]
            detect_video(new_obj, video_path, output_path)
        else:
            detect_video(new_obj, video_path)
    except: pass

    to_karthik=new_obj.final_count
    to_karthik_numpy=np.array(to_karthik)
    to_karthik_df=pd.DataFrame(to_karthik_numpy)
    to_karthik_df.to_csv("final_counts-1.csv")
