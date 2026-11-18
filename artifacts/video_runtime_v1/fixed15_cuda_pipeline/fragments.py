"""Materialize each SM80 MMA B fragment in consecutive lanes in page/head order."""
import torch
def pack(inputs, value):
    x=inputs.reshape(-1,64,12,128).permute(0,2,1,3)
    k,n,l,half,j=torch.meshgrid(
        torch.arange(4 if value else 8,device=x.device),torch.arange(8 if value else 4,device=x.device),
        torch.arange(32,device=x.device),torch.arange(2,device=x.device),torch.arange(4,device=x.device),indexing='ij')
    red=16*k+2*(l%4)+8*(j//2)+j%2
    col=16*n+8*half+l//4
    return x[:,:,red if value else col,col if value else red].contiguous()
