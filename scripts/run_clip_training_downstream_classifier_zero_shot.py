import os

if not os.path.exists('./log/clip/'):
    os.makedirs('./log/clip/')


def eval_zero_shot(gpu, encoder_usage_info, shadow_dataset, downstream_dataset, reference_file, reference_label, trigger_file):
    cmd = f"nohup python3 -u zero_shot.py \
    --encoder_usage_info {encoder_usage_info} \
    --shadow_dataset {shadow_dataset} \
    --reference_file ./reference/CLIP/{reference_file}.npz \
    --dataset {downstream_dataset} \
    --encoder ./output/CLIP/backdoor/{reference_file}/model_200.pth \
    --trigger_file ./trigger/{trigger_file} \
    --reference_label {reference_label} \
    --gpu {gpu} \
    >./log/clip/zero_shot_{downstream_dataset}_{reference_file}_{reference_label}.txt &"

    os.system(cmd)


def eval_zero_shot_clean(gpu, encoder_usage_info, shadow_dataset, downstream_dataset, reference_file, reference_label, trigger_file):
    cmd = f"nohup python3 -u zero_shot.py \
    --encoder_usage_info {encoder_usage_info} \
    --shadow_dataset {shadow_dataset} \
    --reference_file ./reference/CLIP/{reference_file}.npz \
    --dataset {downstream_dataset} \
    --encoder ./output/CLIP/clean_encoder/encode_image.pth \
    --trigger_file ./trigger/{trigger_file} \
    --reference_label {reference_label} \
    --gpu {gpu} \
    >./log/clip/zero_shot_clean_{downstream_dataset}_{reference_file}_{reference_label}.txt &"

    os.system(cmd)


eval_zero_shot(2, 'CLIP', 'cifar10', 'stl10', 'truck', 9, 'trigger_pt_white_173_50_ap_replace.npz')
eval_zero_shot(3, 'CLIP', 'cifar10', 'gtsrb', 'stop', 14, 'trigger_pt_white_173_50_ap_replace.npz')
eval_zero_shot(4, 'CLIP', 'cifar10', 'svhn', 'zero', 0, 'trigger_pt_white_173_50_ap_replace.npz')

eval_zero_shot_clean(2, 'CLIP', 'cifar10', 'stl10', 'truck', 9, 'trigger_pt_white_173_50_ap_replace.npz')
eval_zero_shot_clean(2, 'CLIP', 'cifar10', 'gtsrb', 'stop', 14, 'trigger_pt_white_173_50_ap_replace.npz')
eval_zero_shot_clean(7, 'CLIP', 'cifar10', 'svhn', 'zero', 0, 'trigger_pt_white_173_50_ap_replace.npz')
