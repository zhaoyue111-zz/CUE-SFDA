python scripts/run_cue_sfda.py train \
    --config configs/high_selection_CSS.yaml \
    --gt-path /data/hilab/datasets/WCH_MRI145/DWI/labels \
    --gpu-id 1

python scripts/run_cue_sfda.py train \
    --config configs/high_selection_SUV.yaml \
    --gt-path /data/hilab/datasets/WCH_MRI145/DWI/labels \
    --gpu-id 1