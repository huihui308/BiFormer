"""
#
# cmd:
    $ cd scripts
    $ rm -rf ./../datasets/tricycle;python3 convert_dataset.py --input_dir=/home/david/dataset/classification/tricycle/haitian_0825 --output_dir=./../datasets/tricycle --json_file=./tricycle_classes.json
#

{

  "forward": ["未知", "有棚", "无棚"],

  "backward": ["未知", "有棚", "无棚", "有箱"],

  "color": ["未知", "黑色", "白色", "灰色", "红色", "橙色", "黄色", "绿色", "蓝色", "紫色", "棕色", "粉色"],

  "purpose": ["未知", "载人", "货运", "快递"],

  "brand": ["未知", "顺丰", "申通", "圆通", "中通", "邮政", "京东", "德邦", "韵达", "百世", "苏宁", "天猫", "极兔", "海皇", "嘉德", "品骏", "丹鸟", "多点",  "天天", "博信达", "宅急送"]

}

"""
import cv2
import scipy
import scipy.io
from tqdm import tqdm
from typing import List
from pathlib import Path
from easydict import EasyDict
import os, sys, math, json, shutil, random, datetime, signal, argparse


TQDM_BAR_FORMAT = '{l_bar}{bar:40}| {n_fmt}/{total_fmt} {elapsed}'


def prRed(skk): print("\033[91m \r>> {}: {}\033[00m" .format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), skk))
def prGreen(skk): print("\033[92m \r>> {}:  {}\033[00m" .format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), skk)) 
def prYellow(skk): print("\033[93m \r>> {}:  {}\033[00m" .format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), skk))
def prLightPurple(skk): print("\033[94m \r>> {}:  {}\033[00m" .format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), skk))
def prPurple(skk): print("\033[95m \r>> {}:  {}\033[00m" .format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), skk))
def prCyan(skk): print("\033[96m \r>> {}:  {}\033[00m" .format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), skk))
def prLightGray(skk): print("\033[97m \r>> {}:  {}\033[00m" .format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), skk)) 
def prBlack(skk): print("\033[98m \r>> {}:  {}\033[00m" .format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), skk))


def term_sig_handler(signum, frame)->None:
    prYellow('\r>> {}: \n\n\n***************************************\n'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    prYellow('\r>> {}: Catched singal: {}\n'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), signum))
    prYellow('\r>> {}: \n***************************************\n'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    sys.stdout.flush()
    os._exit(0)
    return


def parse_args(args = None):
    """ parse the arguments. """
    parser = argparse.ArgumentParser(description = 'Convert dataset to torch format.')
    parser.add_argument(
        "--input_dir",
        type = str,
        required = True,
        help = "Input directory to OpenALPR's benchmark end2end us license plates."
    )
    parser.add_argument(
        "--output_dir",
        type = str,
        required = True,
        help = "Ouput directory to resized images/labels."
    )
    parser.add_argument(
        "--json_file",
        type = str,
        required = True,
        help = "Json label file."
    )
    return parser.parse_args(args)


def make_ouput_dir(output_dir:str, class_cnt:int)->None:
    #if os.path.exists(output_dir):
    #    shutil.rmtree(output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for lop_dir0 in ("train", "val"):
        first_dir = os.path.join(output_dir, lop_dir0)
        if not os.path.exists(first_dir):
            #shutil.rmtree(first_dir)
            os.makedirs(first_dir)
        for i in range(class_cnt):
            second_dir = os.path.join(first_dir, 'class' + str(i + 1))
            if not os.path.exists(second_dir):
                os.makedirs(second_dir)
    return


def deal_files(files_list, output_dir, save_dir_dict, statistic_dict)->None:
    pbar = enumerate(files_list)
    pbar = tqdm(pbar, total=len(files_list), desc="Processing", colour='blue', bar_format=TQDM_BAR_FORMAT)
    for (lop_cnt, image_file) in pbar:
        file_name, file_type = os.path.splitext(image_file)
        json_file = file_name + '.json'
        if not Path( json_file ).is_file():
            #prRed('{} not exist, continue'.format(json_file))
            continue
        with open(json_file, "r") as fp:
            json_data = json.load(fp, encoding='utf-8')
            #print(json_data['forward']['name'])
            #print("\n\n")
            #print(json_data)
            labels_list = []
            for label_str in json_data:
                #print(label_str)
                if label_str == 'locate':
                    x0, y0, x1, y1 = json_data['locate']
                    x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
                    if (y0 > y1) or (x0 > x1):
                        cropped_img = None
                    else:
                        img = cv2.imread(image_file)
                        cropped_img = img[y0:y1, x0:x1]  # 裁剪坐标为[y0:y1, x0:x1]
                        #cv2.imwrite("./cv_cut_thor.jpg", cropped_img)
                else:
                    #print(save_dir_dict[label_str][json_data[label_str]['name']])
                    if label_str not in save_dir_dict:
                        continue
                    if json_data[label_str]['name'] not in save_dir_dict[label_str]:
                        continue
                    labels_list.append( save_dir_dict[label_str][json_data[label_str]['name']] )
                    statistic_dict[label_str][json_data[label_str]['name']] += 1
        #print(len(labels_list), output_dir)
        save_file = None
        save_file_name = file_name.split('/')[-1] + "_" + str(random.randint(0, 999999999999)).zfill(12) + file_type
        if (lop_cnt % 10) >= 8:
            save_file = os.path.join(output_dir, 'val')
        else:
            save_file = os.path.join(output_dir, 'train')
        #save_file = os.path.join(save_file, save_file_name)
        #print(save_file)
        if cropped_img is not None:
            for lop_dir in labels_list:
                tmp_save_dir = os.path.join(save_file, lop_dir)
                #tmp_save_file = save_file + '/' + lop_dir + '/' + save_file_name
                #print(tmp_save_file)
                if not os.path.exists(tmp_save_dir):
                    os.makedirs(tmp_save_dir)
                resized = cv2.resize(cropped_img, (224, 224), interpolation = cv2.INTER_AREA)
                cv2.imwrite(os.path.join(tmp_save_dir, save_file_name), resized)
    return


def main_func(args = None):
    """ Main function for data preparation. """
    signal.signal(signal.SIGINT, term_sig_handler)
    args = parse_args(args)
    args.input_dir = os.path.abspath(args.input_dir)
    prYellow('input_dir: {}'.format(args.input_dir))
    args.output_dir = os.path.abspath(args.output_dir)
    prYellow('output_dir: {}'.format(args.output_dir))

    prGreen('json_file: {}'.format(args.json_file))
    with open(args.json_file) as json_file:
        json_contents = json_file.read()
    prGreen('The json file contents is \n{}\n'.format(json_contents))
    label_dict = json.loads(json_contents)
    #print(label_dict, type(label_dict))
    label_cnt = 0
    save_dir_dict, statistic_dict = {}, {}
    for label_key in label_dict:
        save_dir_dict[label_key] = {}
        statistic_dict[label_key] = {}
        for label_val in label_dict[label_key]:
            save_dir_dict[label_key][label_val] = 'class' + str(label_cnt + 1)
            statistic_dict[label_key][label_val] = 0
            label_cnt += 1
    prGreen('Total label count is {}'.format(label_cnt))
    prGreen('save_dir_dict is:\n{}'.format(save_dir_dict))
    prGreen('statistic_dict is:\n{}'.format(statistic_dict))

    files_list = []
    make_ouput_dir(args.output_dir, label_cnt)
    for root, dirs, files in os.walk(args.input_dir):
        for one_file in files:
            file_name, file_type = os.path.splitext(one_file)
            #if file_type != '.json':
            if file_type not in ('.jpg', '.png', '.bmp'):
                continue
            files_list.append( os.path.join(root, one_file) )
    #print(len(files_list), files_list[0])
    print("\n")

    deal_files(files_list, args.output_dir, save_dir_dict, statistic_dict)
    #prGreen('statistic_dict is {}'.format(statistic_dict))

    print("\n")
    prYellow('The result:')
    for label_key in statistic_dict:
        print('{}:'.format(label_key))
        for label_val in statistic_dict[label_key]:
            print("%3s " %(label_val), end="")
        print("\n")
        for label_val in statistic_dict[label_key]:
            print("%5d " %(statistic_dict[label_key][label_val]), end="")
        print("\n")
    print("\n")
    return


if __name__ == "__main__":
    main_func()