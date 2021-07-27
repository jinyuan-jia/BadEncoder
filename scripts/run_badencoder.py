import os

if not os.path.exists('./log/bad_encoder'):
    os.makedirs('./log/bad_encoder')

def run_finetune(gpu, encoder_usage_info, shadow_dataset, downstream_dataset, trigger, reference, clean_encoder='model_1000.pth'):

    save_path = f'./output/{encoder_usage_info}/{downstream_dataset}_backdoored_encoder'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    cmd = f'nohup python3 -u badencoder.py \
    --lr 0.001 \
    --batch_size 256 \
    --results_dir {save_path}/ \
    --shadow_dataset {shadow_dataset} \
    --pretrained_encoder ./output/{encoder_usage_info}/clean_encoder/{clean_encoder} \
    --encoder_usage_info {encoder_usage_info} \
    --gpu {gpu} \
    --reference_file ./reference/{encoder_usage_info}/{reference}.npz \
    --trigger_file ./trigger/{trigger} \
    > ./log/bad_encoder/{encoder_usage_info}_{downstream_dataset}_{reference}.log &'
    os.system(cmd)



run_finetune(0, 'cifar10', 'cifar10', 'stl10', 'trigger_pt_white_21_10_ap_replace.npz', 'truck')
# run_finetune(1, 'cifar10', 'cifar10', 'gtsrb', 'trigger_pt_white_21_10_ap_replace.npz', 'priority')
# run_finetune(2, 'cifar10', 'cifar10', 'svhn', 'trigger_pt_white_21_10_ap_replace.npz', 'one')
