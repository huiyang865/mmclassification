import argparse
import glob
import os
import random

TRAIN_BALANCE = 1000
OTHER_BALANCE = 500


def parse_args():
    parser = argparse.ArgumentParser(description='Making lst file')
    parser.add_argument(
        '--file_dir', type=str, default='', help='the input directory of imgs')
    parser.add_argument(
        '--lst_dir',
        type=str,
        default='',
        help='the output directory of lst files')
    parser.add_argument(
        '--r_dir',
        type=list,
        default=[],
        help='sub directory list for training images')

    args = parser.parse_args()
    return args


def getNextdir(file_dir):
    for _, direnames, _ in os.walk(file_dir):
        return direnames


def getFilename(file_dir):
    num = TRAIN_BALANCE
    if 'all_targets' not in file_dir:
        num = OTHER_BALANCE

    wfilename = []
    xfilename = []
    yfilename = []
    vfilename = []
    for root, _, _ in os.walk(file_dir):
        file = glob.glob(root + '/*.jpg') + glob.glob(root + '/*.png')
        for f in file:
            # 去除新增数据，仅做新品建模
            if 'hard' in f:
                wfilename.append(f)
            else:
                if 'shelf' in f:
                    xfilename.append(f)
                elif 'video' in f:
                    vfilename.append(f)
                else:
                    yfilename.append(f)
    if len(xfilename) < 200:
        filename = wfilename + xfilename + xfilename + yfilename
    else:
        filename = wfilename + xfilename + yfilename

    if len(filename) > num:
        if '_check' in file_dir:
            filename = vfilename[:30] + filename
        n = num
    else:
        filename = filename + vfilename
        n = min(len(filename), num)
    return filename[:n], num


def main():
    args = parse_args()
    label, class_name, num_subject = [], [], []
    index_train, index_val, j = 0, 0, 0
    LST_DIR, file_dir, Rdirs = args.lst_dir, args.file_dir, args.r_dir

    if not os.path.exists(LST_DIR):
        os.makedirs(LST_DIR)

    train_lst_file = LST_DIR + '/train.lst'
    val_lst_file = LST_DIR + '/val.lst'
    info_file = LST_DIR + '/label.inf'

    tf = open(train_lst_file, 'a+')
    vf = open(val_lst_file, 'a+')

    for rdir in Rdirs:
        direnames = getNextdir(file_dir + rdir)
        for i, dir in enumerate(direnames):
            filename, num = getFilename(os.path.join(file_dir, rdir, dir))
            if len(filename) < 1:
                continue
            label.append(j)
            class_name.append(dir)
            num_subject.append(len(filename))
            val_num = int(len(filename) * 0.05)
            if val_num == 0:
                val_num = 1
            random.shuffle(filename)

            if len(filename) >= num:
                xy = 1
            else:
                xy = int(num / float(len(filename))) + 3
            count = 0
            for x in range(xy):
                for line in filename[:0 - val_num]:
                    if count > num:
                        break
                    line = line.replace('', '')
                    tf.write(
                        str(index_train) + '\t' + str(j) + '\t' + line + '\n')
                    index_train += 1
                    count += 1
            if 'others' not in rdir:  # True:#'other' not in rdir:
                for line in filename[0 - val_num:]:
                    line = line.replace('', '')
                    vf.write(
                        str(index_val) + '\t' + str(j) + '\t' + line + '\n')
                    index_val += 1
            j += 1

        print(rdir)
        print(index_train, index_val, j)

    print(index_train, index_val, j)
    with open(info_file, 'a+') as wf:
        for i in range(len(label)):
            wf.write(
                str(label[i]) + '\t' + class_name[i] + '\t' +
                str(num_subject[i]) + '\n')
