import os

if not os.path.exists('./log/cifar10'):
    os.makedirs('./log/cifar10')

def run_eval(gpu, encoder_usage_info, downstream_dataset, encoder, reference_label, trigger, reference_file, key='clean'):
    cmd = f"nohup python3 -u training_downstream_classifier.py \
            --dataset {downstream_dataset} \
            --trigger_file {trigger} \
            --encoder {encoder} \
            --encoder_usage_info {encoder_usage_info} \
            --reference_label {reference_label} \
            --reference_file ./reference/{encoder_usage_info}/{reference_file}.npz \
            --gpu {gpu} \
            >./log/{encoder_usage_info}/evaluation_{key}_{encoder_usage_info}_{downstream_dataset}.txt &"

    os.system(cmd)



run_eval(0, 'cifar10', 'stl10', 'output/cifar10/clean_encoder/model_1000.pth', 9, './trigger/trigger_pt_white_21_10_ap_replace.npz', 'truck')
run_eval(1, 'cifar10', 'gtsrb', 'output/cifar10/clean_encoder/model_1000.pth', 12, './trigger/trigger_pt_white_21_10_ap_replace.npz', 'priority')
run_eval(2, 'cifar10', 'svhn', 'output/cifar10/clean_encoder/model_1000.pth', 1, './trigger/trigger_pt_white_21_10_ap_replace.npz', 'one')

run_eval(3, 'cifar10', 'stl10', './output/cifar10/stl10_backdoored_encoder/model_200.pth', 9, './trigger/trigger_pt_white_21_10_ap_replace.npz', 'truck', 'backdoor')
run_eval(4, 'cifar10', 'gtsrb', './output/cifar10/gtsrb_backdoored_encoder/model_200.pth', 12, './trigger/trigger_pt_white_21_10_ap_replace.npz', 'priority', 'backdoor')
run_eval(5, 'cifar10', 'svhn', './output/cifar10/svhn_backdoored_encoder/model_200.pth', 1, './trigger/trigger_pt_white_21_10_ap_replace.npz', 'one', 'backdoor')
