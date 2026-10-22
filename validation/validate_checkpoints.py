"""Optional inference-only checkpoint validation, adapted from the reviewed script.

External original PNGs, trusted weights, configurations, and archived CSVs are
required. --check-inputs uses only the standard library and checks all hashes.
No training path is present. All execution results go to a fresh output folder.
"""
import argparse
import ast
import csv
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace

from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]


def check_inputs(inputs, archive, stage, evidence=None):
    """Fail before model imports on missing, changed, or incomplete inputs."""
    if evidence is None:
        evidence = json.loads((ROOT / 'validation/evidence/checkpoint-evidence.json').read_text(encoding='utf-8'))
    expected = evidence['external_inputs_sha256']
    checked = {}
    for key, digest in expected.items():
        family, relative = key.split('/', 1)
        if family == 'inputs':
            if stage == 'cnn' and relative != 'cnn_best.pth':
                continue
            if stage == 'freeze' and relative == 'cnn_best.pth':
                continue
        path = (archive if family == 'archive' else inputs) / relative
        if not path.is_file():
            raise ValueError(f'Missing input: {family}/{relative}; see validation/README.md')
        actual = sha(path)
        if actual != digest:
            raise ValueError(f'SHA-256 mismatch: {family}/{relative}')
        checked[key] = actual
    ids = (archive / 'dish_ids_test.txt').read_text(encoding='utf-8').splitlines()
    if len(ids) != 507 or len(set(ids)) != 507:
        raise ValueError('Expected exactly 507 unique test IDs')
    records = []
    for dish in ids:
        if not dish.startswith('dish_') or not dish[5:].isdigit():
            raise ValueError('Invalid test dish ID')
        image = inputs / 'images' / dish / 'rgb.png'
        if not image.is_file():
            raise ValueError(f'Missing original PNG: images/{dish}/rgb.png')
        records.append((dish, sha(image)))
    collection = ''.join(f'{dish} {digest}\n' for dish, digest in sorted(records))
    collection_hash = hashlib.sha256(collection.encode('utf-8')).hexdigest()
    if collection_hash != evidence['source_png_collection_sha256']:
        raise ValueError('Original PNG collection SHA-256 differs from reviewed inputs')
    return {'status': 'PASS', 'input_sha256': checked, 'source_png_collection_sha256': collection_hash}


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def indexed(path):
    with Path(path).open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    result = {r['dish_id']:r for r in rows}
    assert len(result) == len(rows), f'Duplicate dish IDs: {path}'
    return result


def original_class(notebook, name, namespace):
    nodes=[]
    for cell in json.loads(Path(notebook).read_text(encoding='utf-8'))['cells']:
        source=''.join(cell['source'])
        if cell['cell_type'] == 'code' and f'class {name}(' in source:
            nodes.extend(n for n in ast.parse(source).body if isinstance(n,ast.ClassDef) and n.name == name)
    assert len(nodes) == 1
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(notebook),'exec'),namespace)
    return namespace[name]


def save_csv(path, rows):
    with Path(path).open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--archive',type=Path,required=True)
    parser.add_argument('--notebook',type=Path,default=ROOT/'notebooks/food_nutrition_captioning.ipynb')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--check-inputs',action='store_true',help='Verify input checksums and stop before importing ML libraries.')
    parser.add_argument('--stage',choices=['cnn','freeze'],required=True)
    parser.add_argument('--limit',type=int,default=0)
    parser.add_argument('--batch-size',type=int,default=8)
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--facts-csv',type=Path,help='Optional alternate facts for an explicitly separate end-to-end check.')
    args=parser.parse_args()
    if args.batch_size < 1 or args.threads < 1 or not 0 <= args.limit <= 507:
        parser.error('Batch size and threads must be positive; limit must be between 0 and 507.')
    try:
        preflight = check_inputs(args.inputs, args.archive, args.stage)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    if args.check_inputs:
        print(json.dumps(preflight, indent=2))
        return
    if args.output is None:
        parser.error('--output must name a fresh directory for inference')
    if not args.notebook.is_file():
        parser.error('Notebook definitions are missing')
    if args.output.exists():
        parser.error('--output already exists; choose a fresh run directory')
    if args.facts_csv is not None and not args.facts_csv.is_file():
        parser.error('Alternate facts CSV is missing')
    global np, Image, torch, nn, torchvision, T, timm
    import numpy as np
    from PIL import Image
    import torch
    import torch.nn as nn
    import torchvision
    import torchvision.transforms as T
    import timm
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    ids=(args.archive/'dish_ids_test.txt').read_text().splitlines()
    assert len(ids)==len(set(ids))==507
    if args.limit: ids=ids[:args.limit]
    transform=T.Compose([T.Resize((224,224)),T.ToTensor(),T.Normalize([.485,.456,.406],[.229,.224,.225])])
    def images(batch):
        tensors=[]
        for dish in batch:
            with Image.open(args.inputs/'images'/dish/'rgb.png') as image:
                tensors.append(transform(image.convert('RGB')))
        return torch.stack(tensors)
    started=time.perf_counter()
    summary={'stage':args.stage,'rows':len(ids),'device':'cpu','threads':args.threads,
             'batch_size':args.batch_size,'executed_at_utc':datetime.now(timezone.utc).isoformat(),
             'input_verification':preflight,
             'notebook_sha256':sha(args.notebook),
             'versions':{'torch':torch.__version__,'torchvision':torchvision.__version__,'timm':timm.__version__},
             'preprocessing':'PIL RGB; torchvision Resize((224,224)) bilinear/antialias default; ToTensor; ImageNet normalization.',
             'training_performed':False}
    metadata=indexed(args.archive/'metadata_clean.csv')
    columns=['calories','protein','fat','carb','mass']
    if args.stage=='cnn':
        checkpoint=args.inputs/'cnn_best.pth'
        ck=torch.load(checkpoint,map_location='cpu',weights_only=True)
        cls=original_class(args.notebook,'NutritionCNN',dict(nn=nn,torch=torch,timm=timm,N_TARGETS=5,CNN_HIDDEN=256,CNN_DROPOUT=.3))
        model=cls(pretrained=False)
        load=model.load_state_dict(ck['model_state'],strict=True)
        model.eval()
        stats=json.loads((args.archive/'target_stats_log.json').read_text())
        assert stats['cols']==columns and stats['transform']=='log1p'
        mean=torch.tensor(stats['mean']);std=torch.tensor(stats['std'])
        archived=indexed(args.archive/'predictions.csv')
        predictions=[];rows=[]
        with torch.inference_mode():
            for start in range(0,len(ids),args.batch_size):
                batch=ids[start:start+args.batch_size]
                tensor=images(batch)
                out=torch.clamp(torch.expm1(model(tensor)*std+mean),min=0)
                assert torch.isfinite(out).all()
                if start==0:
                    repeat=torch.clamp(torch.expm1(model(tensor)*std+mean),min=0)
                    summary['first_batch_repeat_max_abs_diff']=float((out-repeat).abs().max())
                for dish,values in zip(batch,out.tolist()):
                    row={'dish_id':dish}
                    for col,value in zip(columns,values):
                        row[f'checkpoint_{col}']=value
                        row[f'archived_{col}']=float(archived[dish][f'pred_{col}'])
                        row[f'measured_{col}']=float(metadata[dish][col])
                    rows.append(row);predictions.append(values)
                print(f'CNN {len(rows)}/{len(ids)} elapsed={time.perf_counter()-started:.1f}s',flush=True)
        p=np.array(predictions);y=np.array([[float(metadata[d][c]) for c in columns] for d in ids])
        old=np.array([[float(archived[d][f'pred_{c}']) for c in columns] for d in ids])
        metrics={}
        for i,col in enumerate(columns):
            metrics[col]={'MAE':float(np.abs(p[:,i]-y[:,i]).mean()),'RMSE':float(np.sqrt(np.square(p[:,i]-y[:,i]).mean())),
                'R2':float(1-np.square(p[:,i]-y[:,i]).sum()/np.square(y[:,i]-y[:,i].mean()).sum()),
                'mean_abs_difference_from_archive':float(np.abs(p[:,i]-old[:,i]).mean()),
                'max_abs_difference_from_archive':float(np.abs(p[:,i]-old[:,i]).max())}
        save_csv(args.output/'cnn_predictions_comparison.csv',rows)
        summary.update(checkpoint_sha256=sha(checkpoint),checkpoint_bytes=checkpoint.stat().st_size,
            strict_state_load=True,checkpoint_epoch=ck['epoch'],checkpoint_val_loss=ck['val_loss'],
            metrics=metrics,all_predictions_close_to_archive=bool(np.allclose(p,old,rtol=1e-4,atol=1e-3)),
            archive_comparison_tolerance={'rtol':1e-4,'atol':1e-3},stats_sha256=sha(args.archive/'target_stats_log.json'))
        # New CNN facts are a separate artifact, never replace original facts.
        import re
        oldfacts=indexed(args.archive/'fact_strings.csv')
        newfacts=[]
        labels={'calories':'calories','protein':'protein','fat':'fat','carb':'carbohydrates','mass':'mass'}
        for dish,values in zip(ids,p):
            fact=oldfacts[dish]['fact_string']
            for col,value in zip(columns,values):
                fact,count=re.subn(r'('+labels[col]+r': )\d+',lambda m:m[1]+str(int(round(float(value)))),fact)
                assert count==1
            newfacts.append({'dish_id':dish,'fact_string':fact})
        save_csv(args.output/'cnn_checkpoint_facts.csv',newfacts)
    else:
        import transformers
        from transformers import ViTModel,ViTConfig,T5ForConditionalGeneration,T5Config,T5Tokenizer,GenerationConfig
        from transformers.modeling_outputs import BaseModelOutput
        checkpoint=args.inputs/'captioner_freeze.pth'
        vit_config=ViTConfig.from_pretrained(args.inputs/'vit',local_files_only=True)
        t5_config=T5Config.from_pretrained(args.inputs/'t5',local_files_only=True)
        # Only configs instantiate the model. All parameters then come from the
        # user's checkpoint with a strict load, avoiding duplicate base weights.
        cls=original_class(args.notebook,'ViTT5Captioner',dict(nn=nn,torch=torch,
            ViTModel=SimpleNamespace(from_pretrained=lambda *a:ViTModel(vit_config)),
            T5ForConditionalGeneration=SimpleNamespace(from_pretrained=lambda *a:T5ForConditionalGeneration(t5_config)),
            VIT_ID='config-only',T5_ID='config-only',BEAM_WIDTH=4,MAX_CAPTION_LEN=96,BaseModelOutput=BaseModelOutput))
        model=cls(freeze_vit=True,use_fact=True)
        state=torch.load(checkpoint,map_location='cpu',weights_only=True)
        model.load_state_dict(state,strict=True)
        del state
        model.t5.generation_config=GenerationConfig.from_pretrained(args.inputs/'t5',local_files_only=True)
        model.eval()
        tokenizer=T5Tokenizer.from_pretrained(args.inputs/'t5',local_files_only=True)
        saved=indexed(args.archive/'test_captions_freeze.csv')
        facts=indexed(args.facts_csv or args.archive/'fact_strings.csv')
        rows=[]
        with torch.inference_mode():
            for start in range(0,len(ids),args.batch_size):
                batch=ids[start:start+args.batch_size]
                encoded=tokenizer([facts[d]['fact_string'] for d in batch],max_length=160,padding='max_length',truncation=True,return_tensors='pt')
                gen=model.generate_caption(images(batch),encoded['input_ids'],encoded['attention_mask'])
                texts=tokenizer.batch_decode(gen,skip_special_tokens=True)
                for dish,text in zip(batch,texts):
                    rows.append({'dish_id':dish,'fact_string':facts[dish]['fact_string'],
                        'ground_truth_caption':saved[dish]['ground_truth_caption'],'generated_caption':text,
                        'archived_caption':saved[dish]['generated_caption'],'exact_match':text==saved[dish]['generated_caption']})
                save_csv(args.output/'freeze_captions_comparison.csv',rows)
                print(f'FREEZE {len(rows)}/{len(ids)} exact={sum(r["exact_match"] for r in rows)} elapsed={time.perf_counter()-started:.1f}s',flush=True)
        summary.update(checkpoint_sha256=sha(checkpoint),checkpoint_bytes=checkpoint.stat().st_size,strict_state_load=True,
            exact_caption_matches=sum(r['exact_match'] for r in rows),facts_source='fresh_cnn_checkpoint' if args.facts_csv else 'historical_archived_facts',
            facts_sha256=sha(args.facts_csv or args.archive/'fact_strings.csv'),
            decoding={'num_beams':4,'max_length':96,'early_stopping':True,'no_repeat_ngram_size':3,'fact_max_length':160},
            tokenizer_class=type(tokenizer).__name__,transformers=transformers.__version__)
    summary['seconds']=time.perf_counter()-started
    summary['execution_status']='completed'
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)
    if args.stage == 'freeze' and not args.facts_csv and summary['exact_caption_matches'] != len(ids):
        raise SystemExit('FAIL: historical-fact captions differ; inspect the new run outputs.')

if __name__=='__main__': main()
