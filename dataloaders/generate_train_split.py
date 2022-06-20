import os
import re
from myutils.read_txt import read


def generate(dataset, split):
    if dataset == "hmdb51":
        assert split in ["split1", "split2", "split3"]
        id_map = read("hmdb_labels.txt")

        output_train = open(f"{dataset}_train_{split}.txt", "a")
        output_val = open(f"{dataset}_val_{split}.txt", "a")
        output_test = open(f"{dataset}_test_{split}.txt", "a")

        dataset_Path = r"D:\DATASET\HMDB51"
        video_Path = r"D:\DATASET\HMDB51\hmdb51_unzip"
        annotation = r"D:\DATASET\HMDB51\testTrainMulti_7030_splits"
        ano_list = os.listdir(annotation)

        count = 0
        for epoch in range(len(ano_list)):
            k = ano_list[epoch]
            if re.search(split, k) is not None:
                with open(os.path.join(annotation, k), "r") as f:
                    video_list = f.readlines()
                for i in range(len(video_list)):
                    video_list[i] = video_list[i].strip().split(" ")

                k = k.split("_")
                video_class = "_".join(k[0:len(k) - 2])
                print(k)
                for name, class_id in video_list:
                    if class_id == "1":
                        output_train.writelines(os.path.join(video_class, name) + " " + str(count) + "\n")
                    elif class_id == "2":
                        output_test.writelines(os.path.join(video_class, name) + " " + str(count) + "\n")
                    else:
                        output_val.writelines(os.path.join(video_class, name) + " " + str(count) + "\n")

                count += 1
    else:
        pass


if __name__ == '__main__':
    generate("hmdb51", split="split1")
