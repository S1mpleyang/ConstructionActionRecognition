import os

method = "tsn"
dataset = "myaction"
classes = 7
samplesize = 224
num_frames = 8
model = "STR_Transformer"

result_path = f"result"
eval_path = result_path+"/eval1215"

################## train ##################

"""myattention"""
at_type = "DTM"

sub_path = f"{model}_{at_type}"
string = f"python main.py --result_path {result_path} --sub_path {sub_path} --model {model} --at_type {at_type} --n_classes {classes} --num_frames {num_frames} --sample_size {samplesize} --learning_rate 1e-5 --multistep_milestones 10 20 --gama 0.2  --n_epochs 25 --dataset {dataset} --optimizer adam --batch_size 1 --n_threads 4 --checkpoint 10 --no_val --tensorboard --seg_method {method}"
# print(string)
# os.system(string)

sub_path = f"{model}_{at_type}"
for k in range(1):
    string = f"python evaluate.py --result_path {eval_path} --sub_path {sub_path} --model {model} --at_type {at_type} --n_classes {classes} --resume_path best.pth --num_frames {num_frames} --sample_size {samplesize} --dataset {dataset} --batch_size 1 --n_threads 4 --seg_method {method}"
    # string = f"python evaluate.py --result_path {eval_path} --sub_path {sub_path} --model {model} --at_type {at_type} --n_classes {classes} --resume_path {os.path.join(result_path, sub_path, 'best.pth')} --num_frames {num_frames} --sample_size {samplesize} --dataset {dataset} --batch_size 1 --n_threads 4 --seg_method {method}"
    print(string)
    os.system(string)




