import os

""" parameters """
method = "tsn"
dataset = "myaction"
classes = 7
samplesize = 224
num_frames = 8
model = "STR_Transformer"

result_path = f"result"
eval_path = result_path+"/eval1215"

at_type = "DTM"
test_epochs = 10

################## test ##################
sub_path = f"{model}_{at_type}"
for k in range(test_epochs):
    string = f"python evaluate.py --result_path {eval_path} --sub_path {sub_path} --model {model} --at_type {at_type} --n_classes {classes} --resume_path best.pth --num_frames {num_frames} --sample_size {samplesize} --dataset {dataset} --batch_size 1 --n_threads 4 --seg_method {method}"
 
    print(string)
    os.system(string)




