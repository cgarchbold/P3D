#Full Conditional ADA 256x256 
#python train.py --outdir=./training_runs --cfg=stylegan2 --data=/scratch/datasets/sg3_ucla_256_full.zip --gpus=2 --batch=32 --gamma=10 --aug=ada --cond=True

#Train from crowdcounting pretrained
#python train.py --outdir=./training_runs --cfg=stylegan3-t --data=/scratch/datasets/sg3_ucla_256_full.zip --gpus=2 --batch=32 --gamma=2 --snap=20 --aug=ada --cond=True --resume=/scratch/Projects/stylegan3/training_runs/00038-stylegan3-t-crowd_sg3_256-gpus2-batch32-gamma2/network-snapshot-003840.pkl

#DP Training
#python train.py --outdir=./training_runs --cfg=stylegan3-t --data=/scratch/datasets/sg3_ucla_256_full.zip --gpus=2 --batch=32 --gamma=2 --snap=20 --aug=ada --cond=True --dp_eps=50 --kimg=500


#Train vgkg unconditional
#python train.py --outdir=./training_runs --cfg=stylegan3-t --data=/scratch/datasets/sg3_vgkg_256.zip --gpus=2 --batch=32 --gamma=2 --snap=20 --aug=ada --resume=/scratch/Projects/stylegan3/training_runs/00059-stylegan3-t-sg3_vgkg_256-gpus2-batch32-gamma2/network-snapshot-000960.pkl

#Train from vgkg pretrained
#python train.py --outdir=./training_runs --cfg=stylegan3-t --data=/scratch/datasets/sg3_ucla_256_full.zip --gpus=2 --batch=32 --gamma=2 --snap=20 --aug=ada --cond=True --resume=/scratch/Projects/stylegan3/training_runs/00062-stylegan3-t-sg3_vgkg_256-gpus2-batch32-gamma2/network-snapshot-002160.pkl
