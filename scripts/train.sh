#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")

# modify these augments if you want to try other datasets, splits or methods
# dataset: ['pascal', 'cityscapes', 'coco']
# method: ['unimatch', 'unimatch_msloss' 'fixmatch', 'fixmatch_msloss', 'supervised', 'supervised_msloss',
#           'unimatch_msloss_wo_img_per', 'unimatch_msloss_wo_fea_per']
# exp: just for specifying the 'save_path'
# split: ['92', '1_16', 'u2pl_1_16', ...]. Please check directory './splits/$dataset' for concrete splits
dataset='pascal'
method='unimatch_msloss'
exp='r101'
split='1_4'
no='312_2'

config=/kaggle/working/EVMatch/configs/${dataset}.yaml
labeled_id_path=/kaggle/working/EVMatch/splits/$dataset/$split/labeled.txt
unlabeled_id_path=/kaggle/working/EVMatch/splits/$dataset/$split/unlabeled.txt
save_path=/kaggle/working/EVMatch/exp/$dataset/$method/$exp/${split}_${no}
# save_path=/root/autodl-nas/exp/$dataset/$method/$exp/${split}_${no}
# save_path=exp/$dataset/$method/$exp/${split}_${no}

mkdir -p $save_path
TORCH_DISTRIBUTED_DETAIL=DEBUG
python -m torch.distributed.run \
    --nproc_per_node=$1 \
    --master_addr=localhost \
    --master_port=$2 \
    /kaggle/working/EVMatch/$method.py \
    --config=$config --labeled-id-path $labeled_id_path --unlabeled-id-path $unlabeled_id_path \
    --save-path $save_path \
    --port $2 2>&1 | tee >awk '!/Value: / {print > "'${save_path}/${now}.log'"}')
    #--port $2 2>&1 | tee >(awk '!/Value: / {print > "'${save_path}/${now}.log'"}')
#    --with_cp \

