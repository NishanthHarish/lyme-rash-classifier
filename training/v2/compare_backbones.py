"""Grouped 5-fold CV of a logistic-regression head on frozen embeddings (3 classes)."""
import json, sys, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
items=json.load(open('manifest.json'))
L={'Negative':0,'Positive':1,'Other':2}
y=np.array([L[i['label']] for i in items]); g=np.array([i['group'] for i in items])
src=np.array([i.get('source','rash') for i in items])
train_mask=src!='flowers'   # flowers never trained on: unseen OOD probe
for name in sys.argv[1:]:
    E=np.load(f'emb_{name}.npy')
    P=np.zeros((len(y),3)); Pf=[]
    idx=np.where(train_mask)[0]
    for tr,te in StratifiedGroupKFold(5,shuffle=True,random_state=0).split(E[idx],y[idx],g[idx]):
        tr,te=idx[tr],idx[te]
        sc=StandardScaler().fit(E[tr]); clf=LogisticRegression(C=0.05,max_iter=3000,class_weight='balanced').fit(sc.transform(E[tr]),y[tr])
        P[te]=clf.predict_proba(sc.transform(E[te])); Pf.append(clf.predict_proba(sc.transform(E[~train_mask])))
    r=(y<2)&train_mask
    pr=P[r]; yr=y[r]
    pos=pr[:,1]/(pr[:,0]+pr[:,1])
    acc=((pos>0.5)==yr).mean(); rec=(pos[yr==1]>0.5).mean(); spec=(pos[yr==0]<=0.5).mean()
    auc=roc_auc_score(yr,pos)
    rej_rash=(pr.argmax(1)==2).mean()
    o=(y==2)&train_mask; rej_ood=(P[o].argmax(1)==2).mean()
    skin=src=='oi_skin'; rej_skin=(P[skin].argmax(1)==2).mean()
    rej_fl=(np.mean(Pf,0).argmax(1)==2).mean()
    print(f'{name:15s} rashAcc {acc:.3f} AUC {auc:.3f} PosRecall {rec:.3f} Spec {spec:.3f} | rash wrongly rejected {rej_rash:.3f} | OOD rejected {rej_ood:.3f} (skin {rej_skin:.3f}, unseen flowers {rej_fl:.3f})')
