import csv, random, collections
random.seed(0)
want={'/m/0dzf4':'arm','/m/035r7c':'leg','/m/0k65p':'hand','/m/031n1':'foot','/m/0dzct':'face','/m/07j2f':'tattoo'}
boxes=collections.defaultdict(list); allids=set()
for r in csv.DictReader(open('ood/validation-annotations-bbox.csv')):
    allids.add(r['ImageID'])
    if r['LabelName'] in want and r['IsGroupOf']=='0' and r['IsDepiction']=='0':
        x0,x1,y0,y1=map(float,(r['XMin'],r['XMax'],r['YMin'],r['YMax']))
        if (x1-x0)*(y1-y0)>0.04: boxes[want[r['LabelName']]].append((r['ImageID'],x0,x1,y0,y1))
quota={'arm':450,'leg':300,'hand':350,'foot':120,'face':200,'tattoo':80}
sel=[]
for k,q in quota.items():
    random.shuffle(boxes[k]); sel+= [(k,)+b for b in boxes[k][:q]]
scene=random.sample(sorted(allids),900)
with open('ood/oi_boxes.csv','w') as f:
    w=csv.writer(f); w.writerows(sel)
ids=sorted({s[1] for s in sel}|set(scene))
open('ood/oi_scene.txt','w').write('\n'.join(scene))
open('ood/oi_ids.txt','w').write('\n'.join(ids))
print(len(sel),len(ids))
