from .finalize import *
import argparse
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--group');p.add_argument('--seed',type=int);a=p.parse_args();torch.set_num_threads(2);dest=OUT/'training'/f'{"A_matched" if a.group=="A" else a.group}_s{a.seed}';ck=torch.load(dest/'selected.pt',map_location='cuda',weights_only=False);bundle=Bundle();bundle.layers.load_state_dict(ck['layers']);bundle.reader.load_state_dict(ck['reader']);rows=json.loads((dest/f'dev_epoch{ck["epoch"]}.json').read_text());_,error=cached_audit(bundle,a.group,a.seed,rows);print('audit',a.group,a.seed,error)
