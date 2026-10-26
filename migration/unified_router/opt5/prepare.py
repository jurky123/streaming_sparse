"""Only prepare tensors consumed by the weight-reuse route."""
import torch
import torch.nn.functional as F

def prepare_weights(prepared,content_energy):
    keys=[];values=[];logs=[];grams=[];starts=[];counts=[];pages=[]
    off=0;page=0
    for key,lc,value in prepared[1]:
        h,b,p=lc.shape
        keys.append(F.pad(key.reshape(h,b,p,64),(0,0,0,16-p)).half())
        if content_energy:values.append(F.pad(value,(0,0,0,16-p)).half())
        else:grams.append(F.pad(value@value.transpose(-1,-2),(0,16-p,0,16-p)))
        logs.append(F.pad(lc,(0,16-p),value=-float('inf')))
        starts.append(torch.arange(b,device=key.device,dtype=torch.int32)*p+off)
        counts.append(torch.full((b,),p,device=key.device,dtype=torch.int32))
        pages.append(torch.arange(page,page+b,device=key.device,dtype=torch.int32).repeat_interleave(p))
        off+=b*p;page+=b
    dummy=prepared[0]
    return dict(padded_key=torch.cat(keys,1),padded_lc=torch.cat(logs,1),
        padded_value=torch.cat(values,1)if content_energy else dummy,
        grams=dummy if content_energy else torch.cat(grams,1),
        starts=torch.cat(starts),counts=torch.cat(counts),page_index=torch.cat(pages),
        value=torch.cat([v.flatten(1,2)for _,_,v in prepared[1]],1))
