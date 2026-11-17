import argparse, collections, hashlib, json, random, shutil
from pathlib import Path
import xml.etree.ElementTree as ET
from PIL import Image

CLASSES = ['dry_knot', 'sound_knot', 'edge_knot']

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--seed', type=int, default=2026)
    args = ap.parse_args()
    root = Path(args.root).resolve()
    out = root / 'DK_SK_EK_COCO'
    if out.exists():
        raise SystemExit(f'Output already exists; refusing to overwrite: {out}')
    samples = []
    counts = collections.Counter()
    for xml in sorted((root / 'Annotations').glob('*.xml')):
        tree = ET.parse(xml)
        img = root / 'JPEGImages' / (xml.stem + '.jpg')
        with Image.open(img) as im:
            w, h = im.size
        objects = []
        for obj in tree.findall('object'):
            name = obj.findtext('name', '').strip()
            if name not in CLASSES:
                continue
            b = obj.find('bndbox')
            # Pascal VOC uses 1-based inclusive coordinates; COCO uses zero-based xywh.
            x1, y1 = float(b.findtext('xmin'))-1, float(b.findtext('ymin'))-1
            x2, y2 = float(b.findtext('xmax')), float(b.findtext('ymax'))
            x1,y1,x2,y2 = max(0,x1),max(0,y1),min(w,x2),min(h,y2)
            if x2 <= x1 or y2 <= y1:
                raise ValueError(f'Invalid box: {xml}')
            objects.append((name,[x1,y1,x2-x1,y2-y1]))
            counts[name] += 1
        if objects:
            samples.append((img,w,h,objects,hashlib.sha256(img.read_bytes()).hexdigest()))
    hashes = [s[4] for s in samples]
    if len(set(hashes)) != len(hashes):
        raise ValueError('Duplicate image bytes found; resolve before splitting to avoid leakage')
    random.Random(args.seed).shuffle(samples)
    n = len(samples); a=int(n*.6); b=a+int(n*.2)
    splits = {'train':samples[:a], 'val':samples[a:b], 'test':samples[b:]}
    (out/'annotations').mkdir(parents=True)
    summary={'seed':args.seed,'source_images':len(list((root/'JPEGImages').glob('*.jpg'))),'selected_images':n,'object_counts':dict(counts),'splits':{}}
    for split, rows in splits.items():
        (out/split).mkdir()
        data={'info':{'description':'OULU-DET DK/SK/EK subset'},'licenses':[], 'images':[], 'annotations':[], 'categories':[{'id':i+1,'name':c} for i,c in enumerate(CLASSES)]}
        split_counts=collections.Counter()
        for iid,(img,w,h,objects,digest) in enumerate(rows,1):
            shutil.copy2(img,out/split/img.name)
            data['images'].append({'id':iid,'file_name':img.name,'width':w,'height':h})
            for name,box in objects:
                split_counts[name]+=1
                data['annotations'].append({'id':len(data['annotations'])+1,'image_id':iid,'category_id':CLASSES.index(name)+1,'bbox':box,'area':box[2]*box[3],'iscrowd':0})
        (out/'annotations'/f'instances_{split}.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
        summary['splits'][split]={'images':len(rows),'objects':dict(split_counts),'files':[r[0].name for r in rows]}
    (out/'manifest.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k!='splits'},ensure_ascii=False))
    print({k:len(v) for k,v in splits.items()});print(out)

if __name__=='__main__':
    main()
