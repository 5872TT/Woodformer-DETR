"""Local OULU training entry; --check checks a real batch without optimizer updates."""
import argparse
import os
from pathlib import Path
import sys
import json

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',choices=['baseline','fewd'],default='baseline')
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=80)
    p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--workers',type=int,default=0)
    p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--resume',type=Path)
    p.add_argument('--output',type=Path)
    p.add_argument('--pretrained',type=Path,help='COCO checkpoint for baseline fine-tuning')
    p.add_argument('--check',action='store_true')
    p.add_argument('--amp',action='store_true')
    a=p.parse_args()
    a.data=a.data.resolve()
    if a.resume:
        a.resume=a.resume.resolve()
        if a.resume.name=='best.pt': p.error('best.pt contains inference weights only; use --pretrained for fine-tuning')
    if a.output: a.output=a.output.resolve()
    os.chdir(ROOT)
    import torch, yaml
    from src.core import YAMLConfig
    from src.misc.dist import set_seed
    from src.solver import TASKS
    if a.epochs<1 or a.batch_size<1 or a.workers<0: p.error('Invalid epochs/batch size/workers')
    for split in ('train','val','test'):
        path=a.data/'annotations'/f'instances_{split}.json'
        data=json.loads(path.read_text(encoding='utf-8'))
        if [(x['id'],x['name']) for x in data['categories']] != [(1,'dry_knot'),(2,'sound_knot'),(3,'edge_knot')]:
            p.error(f'Unexpected category mapping: {path}')
        if not data['images']: p.error(f'Empty split: {path}')
        for im in data['images']:
            if not (a.data/split/im['file_name']).is_file(): p.error(f'Missing image: {im}')
    base=ROOT/'configs/rtdetr/fewd_rtdetr_r18_ouludefect.yml'
    pretrained = (a.pretrained or ROOT.parents[1]/'weights/best.pt').resolve() if a.model=='baseline' else None
    if pretrained and not a.resume and not pretrained.is_file():
        p.error(f'Pretrained checkpoint missing: {pretrained}')
    output=a.output or ROOT/'output'/f'oulu_{a.model}{"_pretrained" if pretrained else ""}'
    config={'__include__':[str(base)],'output_dir':str(output),'epoches':a.epochs,'sync_bn':False,'use_amp':a.amp,
            'optimizer':{'params':[], 'lr':0.0001},'lr_scheduler':{'milestones':[max(1,int(a.epochs*.8))]}}
    for split in ('train','val'):
        config[f'{split}_dataloader']={'batch_size':a.batch_size,'num_workers':a.workers,
          'dataset':{'img_folder':str(a.data/split),'ann_file':str(a.data/'annotations'/f'instances_{split}.json')}}
    if a.model=='baseline':
        config['optimizer']['params']=[{'params':'backbone','lr':0.00001}]
        config.update({'RTDETR':{'backbone':'PResNet','encoder':'HybridEncoder'},
          'PResNet':{'depth':18,'variant':'d','return_idx':[1,2,3],'freeze_at':-1,'freeze_norm':False,'pretrained':False},
          'HybridEncoder':{'in_channels':[128,256,512],'hidden_dim':256,'expansion':0.5,'eval_spatial_size':[640,640]},
          'RTDETRTransformer':{'num_decoder_layers':3}})
    config_dir=ROOT/'configs/local';config_dir.mkdir(exist_ok=True)
    path=config_dir/f'oulu_{a.model}.yml';path.write_text(yaml.safe_dump(config,allow_unicode=True,sort_keys=False),encoding='utf-8')
    set_seed(a.seed)
    cfg=YAMLConfig(str(path),resume=str(a.resume) if a.resume else None,
                   tuning=str(pretrained) if pretrained and not a.resume else None)
    print(f'Config: {path}\nDevice: {cfg.device}\nOutput: {output}',flush=True)
    if a.check:
        # No training loop, optimizer, backward, or checkpoint updates.
        cfg.model.eval().to(cfg.device)
        images,targets=next(iter(cfg.train_dataloader))
        with torch.inference_mode(): result=cfg.model(images.to(cfg.device))
        targets=[{k:v.to(cfg.device) if isinstance(v,torch.Tensor) else v for k,v in t.items()} for t in targets]
        with torch.inference_mode():
            losses=cfg.criterion.to(cfg.device)(result,targets)
        if not all(torch.isfinite(v).all() for v in losses.values()):
            raise ValueError('Non-finite losses in preflight')
        next(iter(cfg.val_dataloader))
        print('Losses', {k:float(v) for k,v in losses.items()})
        print('Batch',tuple(images.shape),'labels',sorted(set(int(x) for t in targets for x in t['labels'])))
        print('Output', {k:tuple(v.shape) for k,v in result.items() if isinstance(v,torch.Tensor)})
        print('CHECK OK: no training performed.')
        return
    if (output/'checkpoint.pth').exists() and not a.resume:
        p.error('Output already contains a checkpoint; use --resume or a different --output')
    TASKS[cfg.yaml_cfg['task']](cfg).fit()

if __name__=='__main__':
    main()
