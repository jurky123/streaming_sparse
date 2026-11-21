"""Pad the compact reader's fixed slots without changing legal swaps."""
import torch
def stratified_pool(risk, mask, own):
    p = risk.shape[-1]
    page = torch.arange(p, device=risk.device)
    movable = mask & (page >= 24) & (page != own[..., None])
    removes, adds = [], []
    for lo, hi in ((24, p - 72), (p - 72, p)):
        score = risk[..., lo:hi]
        removes.append(score.masked_fill(~movable[..., lo:hi], torch.inf).argsort(dim=-1, stable=True)[..., :8] + lo)
        adds.append(score.masked_fill(mask[..., lo:hi], -torch.inf).argsort(dim=-1, descending=True, stable=True)[..., :16] + lo)
    r, a = torch.cat(removes, -1), torch.cat(adds, -1)
    legal = movable.gather(-1, r)[..., :, None] & ~mask.gather(-1, a)[..., None, :]
    legal = legal & ((r[..., :, None] >= p - 72) == (a[..., None, :] >= p - 72))
    return r, a, legal




def pool(risk, mask, own):
    if risk.shape[-1]==72:
        page=torch.arange(72,device=risk.device)
        movable=mask & (page>=24) & (page!=own[...,None])
        # Use distinct exact anchor pages as inactive removals and distinct
        # omitted pages as inactive additions. Only actual current swaps are legal.
        order=risk.masked_fill(~movable,torch.inf).argsort(dim=-1,stable=True)
        anchors=((page>=24)|(page==own[...,None])).int().argsort(dim=-1,stable=True)
        slot=torch.arange(16,device=risk.device).expand(*risk.shape[:2],16)
        used=movable.sum(-1,keepdim=True).clamp_max(8)
        remove=torch.where(slot<used,order[...,:16],anchors.gather(-1,(slot-used).clamp_min(0)))
        add=risk.masked_fill(mask,-torch.inf).argsort(dim=-1,descending=True,stable=True)[...,:32]
        legal=(slot<used)[...,None] & (torch.arange(32,device=risk.device)<16)
        return remove,add,legal
    remove, add, legal = stratified_pool(risk, mask, own)
    # Anchor pages are selected and individually distinct. They fill missing
    # removal slots for aggregation only, and are excluded by the original legal mask.
    valid_remove = mask.gather(-1, remove) & (remove != own[..., None])
    anchor = torch.arange(remove.size(-1), device=risk.device)
    remove = torch.where(valid_remove, remove, anchor)
    # Omitted pages outside the add list are equivalent background terms when
    # used as inactive additions. The reader needs each slot to own a unique page.
    used = mask.scatter(-1, add, True)
    pages = torch.arange(mask.size(-1),device=risk.device)
    spare = torch.where(used,mask.size(-1),pages).topk(add.size(-1),largest=False).values
    add = torch.where(mask.gather(-1, add),spare,add)
    return remove, add, legal
