#!/usr/bin/env python3
"""Map SAT output masks back to VoxTell's reoriented target grid."""
import argparse, csv
from pathlib import Path
import importlib.machinery
if not hasattr(importlib.machinery.FileFinder, 'find_module'):
    importlib.machinery.FileFinder.find_module=lambda self,fullname:self.find_spec(fullname).loader
import monai, nibabel as nib, numpy as np
from cue_sfda.data import load_split
from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient
def sat_path(root, case, dataset='D_no_response'):
    stem=case[:-7] if case.endswith('.nii.gz') else Path(case).stem
    p=root/dataset/f'seg_{stem}'/'liver cancer.nii.gz'
    return p if p.is_file() else root/dataset/f'seg_{stem}.nii.gz'

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--image-dir',required=True); ap.add_argument('--split-json',required=True); ap.add_argument('--routing-csv',required=True); ap.add_argument('--sat-root',required=True); ap.add_argument('--output-dir',required=True); ap.add_argument('--dataset',default='D_no_response'); args=ap.parse_args()
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    tr,te=load_split(args.image_dir,args.split_json); cases={c.name:c for c in tr+te}
    ids=[]
    with open(args.routing_csv,encoding='utf-8-sig') as f:
        ids=[r['case_id'] for r in csv.DictReader(f) if r.get('response_state')=='No-response']
    reader=NibabelIOWithReorient()
    tx=monai.transforms.Compose([monai.transforms.LoadImaged(keys=['image']),monai.transforms.EnsureChannelFirstd(keys=['image']),monai.transforms.Orientationd(axcodes='RAS',keys=['image']),monai.transforms.Spacingd(keys=['image'],pixdim=(1,1,3),mode='bilinear'),monai.transforms.CropForegroundd(keys=['image'],source_key='image'),monai.transforms.ToTensord(keys=['image'])])
    for case in ids:
        image_path=cases[case].image_path; sat=sat_path(Path(args.sat_root),case,args.dataset)
        if not sat.is_file(): raise FileNotFoundError(sat)
        sat_arr=np.asarray(nib.load(str(sat)).dataobj)>0.5
        # Derive the exact SAT->raw-index map by transforming coordinate fields
        # through the same MONAI pipeline used by SAT. This avoids orientation
        # ambiguities between SAT's saved identity-affine output and nnUNet.
        raw_img=nib.load(str(image_path)); tgt_shape=tuple(int(x) for x in raw_img.shape); tgt_aff=np.asarray(raw_img.affine)
        coord=[]
        for axis in range(3):
            tmpc=out/f'.coord_{case}_{axis}.nii.gz'
            grid=np.indices(tgt_shape,dtype=np.float32)[axis]
            nib.save(nib.Nifti1Image(grid,tgt_aff),str(tmpc))
            c_tx=monai.transforms.Compose([monai.transforms.LoadImaged(keys=['image','coord']),monai.transforms.EnsureChannelFirstd(keys=['image','coord']),monai.transforms.Orientationd(axcodes='RAS',keys=['image','coord']),monai.transforms.Spacingd(keys=['image','coord'],pixdim=(1,1,3),mode=('bilinear','bilinear')),monai.transforms.CropForegroundd(keys=['image','coord'],source_key='image'),monai.transforms.ToTensord(keys=['image','coord'])])
            coord.append(c_tx({'image':str(image_path),'coord':str(tmpc)})['coord'].numpy()[0]); tmpc.unlink()
        sat_shape=coord[0].shape; common=tuple(min(sat_shape[i],sat_arr.shape[i]) for i in range(3)); mask=np.zeros(sat_shape,bool); mask[:common[0],:common[1],:common[2]]=sat_arr[:common[0],:common[1],:common[2]]
        mapped=np.zeros(tgt_shape,dtype=np.uint8)
        for c in zip(*(x[mask] for x in coord)):
            idx=tuple(int(round(v)) for v in c)
            if all(0<=idx[i]<tgt_shape[i] for i in range(3)): mapped[idx]=1
        tmp=out/f'.tmp_{case}'; nib.save(nib.Nifti1Image(mapped,tgt_aff),str(tmp)+'.nii.gz')
        reader_arr,_=reader.read_seg(str(tmp)+'.nii.gz')
        np.save(out/f'{case}.npy', (reader_arr>0).astype(np.uint8)); Path(str(tmp)+'.nii.gz').unlink()
        print(case, sat_arr.sum(), mapped.sum(), int((reader_arr>0).sum()), tuple(reader_arr.shape), flush=True)
if __name__=='__main__': main()
