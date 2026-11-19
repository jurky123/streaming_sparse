from pathlib import Path
from paths import RUN
cache=RUN/'cache'
files={
'imaging_quality':'pyiqa_model/musiq_spaq_ckpt-358bb6af.pth',
'appearance_style':'clip_model/ViT-B-32.pt','background_consistency':'clip_model/ViT-B-32.pt',
'aesthetic_quality':'clip_model/ViT-L-14.pt',
'overall_consistency':'ViCLIP/ViClip-InternVid-10M-FLT.pth','temporal_style':'ViCLIP/ViClip-InternVid-10M-FLT.pth',
'subject_consistency':'dino_model/dino_vitbase16_pretrain.pth','motion_smoothness':'amt_model/amt-s.pth',
'dynamic_degree':'raft_model/models/raft-things.pth','chunk_discontinuity':'raft_model/models/raft-things.pth',
'human_action':'umt_model/l16_ptk710_ftk710_ftk400_f16_res224.pth','scene':'caption_model/tag2text_swin_14m.pth'}
for d in ('color','spatial_relationship','object_class','multiple_objects'):files[d]='grit_model/grit_b_densecap_objectdet.pth'
def ready(d):
 if d not in files:return True
 p=cache/files[d]
 if p.with_suffix('.verified').is_file():return True
 if p.with_suffix('.download').exists():return False
 return p.is_file() and p.stat().st_size>1000000
