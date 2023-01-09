import os

method = "tsn"
dataset = "myaction"
classes = 7
samplesize = 224
num_frames = 8
model = "STR_Transformer"

result_path = f"result"
eval_path = result_path+"/eval1215"

################## test ##################

at_type = "DTM"

sub_path = f"{model}_{at_type}"
for k in range(10):
    # string = f"python evaluate.py --result_path {eval_path} --sub_path {sub_path} --model {model} --at_type {at_type} --n_classes {classes} --resume_path best.pth --num_frames {num_frames} --sample_size {samplesize} --dataset {dataset} --batch_size 1 --n_threads 4 --seg_method {method}"
    string = f"python evaluate.py --result_path {eval_path} --sub_path {sub_path} --model {model} --at_type {at_type} --n_classes {classes} --resume_path {os.path.join(result_path, sub_path, 'best.pth')} --num_frames {num_frames} --sample_size {samplesize} --dataset {dataset} --batch_size 1 --n_threads 4 --seg_method {method}"
    print(string)
    os.system(string)




