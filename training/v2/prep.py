"""Dedup rash images (exact md5, then near-duplicates by perceptual hash) and write a manifest
with group ids so near-duplicates always land in the same split."""
import hashlib, json, sys
from collections import defaultdict
from pathlib import Path
from PIL import Image
import imagehash
DATA=Path('data/dataset')
by_hash=defaultdict(set); first={}
for label in ['Negative','Positive']:
    for p in sorted((DATA/label).glob('*.jpg')):
        h=hashlib.md5(p.read_bytes()).hexdigest(); by_hash[h].add(label); first.setdefault(h,p)
items=[(str(first[h]), next(iter(ls))) for h,ls in by_hash.items() if len(ls)==1]
print('exact-unique', len(items), 'conflicts', sum(len(l)>1 for l in by_hash.values()))
ph=[imagehash.phash(Image.open(p).convert('RGB')) for p,_ in items]
parent=list(range(len(items)))
def f(i):
    while parent[i]!=i: parent[i]=parent[parent[i]]; i=parent[i]
    return i
pairs=0
for i in range(len(items)):
    for j in range(i+1,len(items)):
        if ph[i]-ph[j]<=8: parent[f(i)]=f(j); pairs+=1
groups=defaultdict(list)
for i in range(len(items)): groups[f(i)].append(i)
# a group with conflicting labels is dropped (same picture labelled both ways)
out=[]; dropped=0
for g,(root,idx) in enumerate(groups.items()):
    labs={items[i][1] for i in idx}
    if len(labs)>1: dropped+=len(idx); continue
    out+=[{'path':items[i][0],'label':items[i][1],'group':f'r{g}'} for i in idx]
print('near-dup pairs', pairs, 'groups', len(groups), 'dropped(conflict)', dropped, 'kept', len(out))
print({l:sum(o['label']==l for o in out) for l in ['Negative','Positive']})
for d in sorted(Path('ood/pool').iterdir()):
    for p in sorted(d.glob('*.jpg')):
        out.append({'path':str(p),'label':'Other','group':'o_'+p.stem.split('_')[1] if d.name=='oi_skin' else 'o_'+p.stem,'source':d.name})
json.dump(out,open('manifest.json','w'),indent=0)
