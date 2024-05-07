import glob
import os
from pathlib import Path

path = r'C:\xukejian\Projects\yolov5-6.2.0\runs\detect\exp9\labels'
p1 = glob.glob(str(Path(path) / '*.txt'), recursive=False)
labels_clas_id = []
for q in p1:
    if os.path.split(q)[1] == "classes.txt":
        continue

    lines = []
    lines2 = []

    with open(q, 'r') as f:
        lines = [x.strip() for x in f.read().strip().splitlines() if len(x.strip())]

    for l in lines:
        b = [float(x) for x in l.split(' ')]
        cid, cx, cy, cw, ch = int(b[0]), b[1], b[2], b[3], b[4]
        if cid == 230:
            cid = 221
            labels_clas_id.append(cid)
            lines2.append('%d %.6f %.6f %.6f %.6f' % (cid, cx, cy, cw, ch))
        elif cid == 140:
            labels_clas_id.append(cid)
            lines2.append('%d %.6f %.6f %.6f %.6f' % (cid, cx, cy, cw, ch))
        # elif cid == 2:
        #     cid = 226
        # elif cid == 3:
        #     cid = 225
        # elif cid == 4:
        #     cid = 211
        # elif cid == 5:
        #     cid = 201
        # labels_clas_id.append(cid)
        # lines2.append('%d %.6f %.6f %.6f %.6f' % (cid, cx, cy, cw, ch))

    with open(q, 'w+') as f:
        for l in lines2:
            f.write(l+'\n')

    print(q, '---', list(set(labels_clas_id)))

print(list(set(labels_clas_id)))