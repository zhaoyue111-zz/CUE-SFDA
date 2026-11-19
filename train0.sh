python scripts/run_cue_sfda.py train \
    --config configs/high_selection_GA.yaml \
    --gt-path /data/hilab/datasets/WCH_MRI145/DWI/labels \
    --gpu-id 0

python scripts/run_cue_sfda.py train \
    --config configs/high_selection_CUV.yaml \
    --gt-path /data/hilab/datasets/WCH_MRI145/DWI/labels \
    --gpu-id 0