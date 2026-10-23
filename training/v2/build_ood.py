"""Build the out-of-scope image pool: Open Images body-part crops (healthy skin),
Open Images scenes, Imagenette objects, TF flowers (held out as unseen OOD test)."""
import csv, random, glob
from pathlib import Path
from PIL import Image
random.seed(0)
O=Path('ood'); OUT=O/'pool'
def save(im, name):
    im=im.convert('RGB'); im.thumbnail((320,320)); p=OUT/name; p.parent.mkdir(parents=True,exist_ok=True); im.save(p,quality=90)
n=0
for k,iid,x0,x1,y0,y1 in csv.reader(open(O/'oi_boxes.csv')):
    try: im=Image.open(O/'oi_raw'/f'{iid}.jpg')
    except Exception: continue
    W,H=im.size; box=(int(float(x0)*W),int(float(y0)*H),int(float(x1)*W),int(float(y1)*H))
    if min(box[2]-box[0],box[3]-box[1])<120: continue
    save(im.crop(box), f'oi_skin/{k}_{iid}_{n}.jpg'); n+=1
for iid in open(O/'oi_scene.txt').read().split():
    try: save(Image.open(O/'oi_raw'/f'{iid}.jpg'), f'oi_scene/{iid}.jpg')
    except Exception: pass
imn=sorted(glob.glob(str(O/'imagenette2-320/*/*/*.JPEG'))); random.shuffle(imn)
for p in imn[:1500]: save(Image.open(p), 'imagenette/'+Path(p).parent.name+'_'+Path(p).name.replace('JPEG','jpg'))
fl=sorted(glob.glob(str(O/'flower_photos/*/*.jpg'))); random.shuffle(fl)
for p in fl[:400]: save(Image.open(p), 'flowers/'+Path(p).parent.name+'_'+Path(p).name)
for d in sorted(OUT.iterdir()): print(d.name, len(list(d.iterdir())))
