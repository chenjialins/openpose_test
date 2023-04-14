import os
import cv2
import sys
import csv
import random
import argparse

# Windows Import pyopenpose
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + '/../../python/openpose/Release')
os.environ['PATH']  = os.environ['PATH'] + ';' + dir_path + '/../../x64/Release;' +  dir_path + '/../../bin;'
import pyopenpose as op

# step1：从视频数据提取图片序列
def Data_Process(data_path):
    # step1:将视频地址存放在列表内
    vedio_path_list = []
    for i in (os.listdir(data_path)):
        path1 = data_path+"/"+i
        for j in (os.listdir(path1)):
            path2 = path1+"/"+j
            vedio_path_list.append(path2)
    # print(vedio_path_list[0])                         # 数据格式：vedio_data/TaiChi/v_TaiChi_g02_c01.avi

    # step2:从每个视频中提取图片序列
    image_folder = "image_data/"
    image_path_list = []
    for vedio_path in vedio_path_list:
        label = vedio_path.split("/")[1]   
        folder = vedio_path.split("/")[2].split(".")[0]
        cap = cv2.VideoCapture(vedio_path)
        frame_fre = cap.get(cv2.CAP_PROP_FRAME_COUNT)//15
        fps_count, index = 0, 0
        while True:
            flag, frame = cap.read()
            if not flag:
                break
            if fps_count % frame_fre == 0 and index < 15:
                index += 1
                path = image_folder+"{}/{}".format(label, folder)
                if not os.path.exists(path):
                    os.mkdir(path)
                image_dir = path+"/{}.jpg".format(index)
                cv2.imwrite(image_dir, frame)
                image_path_list.append(image_dir)
            fps_count += 1

    # step3:保存图片地址
    with open("images_path.csv", "w", newline="") as f:
        writer = csv.writer(f)
        for i in image_path_list:
            writer.writerow([i])
    print("完成视频提取图片并保存")


# step2:提取图片的关键点
def keypoints_from_images(images_path):
    # Flags
    parser = argparse.ArgumentParser()
    args = parser.parse_known_args()

    # Custom Params
    params = dict()
    params["model_folder"] = "../../../models/"
    params["net_resolution"] = "160x160"
    params["disable_blending"] = True

    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    # Process and save images
    with open(images_path, "r", newline="") as f:
        reader = csv.reader(f)
        for [i] in reader:
            datum = op.Datum()
            datum.cvInputData = cv2.imread(i)
            opWrapper.emplaceAndPop(op.VectorDatum([datum]))
            cv2.imwrite(i, datum.cvOutputData)
    print("完成图像序列人体关键点提取")

# step3:制作数据集的name和label
def dataset_path(path):
    dataset_name_list, dataset_label_list = [], []
    for i in (os.listdir(path)):
        dir1 = path+"/"+i
        for j in (os.listdir(dir1)):
            dir2 = dir1+"/"+j
            dataset_name_list.append(dir2)
    random.shuffle(dataset_name_list)
    
    with open("dataset_list.csv", "w", newline="") as f:
        writer = csv.writer(f)
        for name in dataset_name_list:
            label = 1
            if name.split("/",2)[1] == "TaiChi":
                label = 0
            writer.writerow([name, label])
    print("done")

if __name__ == "__main__":
    # step1 test
    # vedio_path = "vedio_data"
    # Data_Process(vedio_path)

    # step2 test
    imageData_csv = "images_path.csv"
    keypoints_from_images(imageData_csv)

    # step3 test
    image_dataset_path = "image_data"
    dataset_path(image_dataset_path)

    pass