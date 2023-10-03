import os
import random
import xml.etree.ElementTree as ET
import numpy as np

classes_path = 'D:\\PycharmProjects\\pythonProject2\\classes.txt'  # 总类别
trainval_percent = 0.9  # 训练集和验证集的比例
train_percent = 0.9


# 获取类别
def get_classes(classes_path):
    with open(classes_path, encoding='utf-8') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names, len(class_names)


VOCdevkit_path = 'D:\\PycharmProjects\\pythonProject2\\PascalVOC2012\\VOCdevkit'
VOCdevkit_sets = [('2012', 'train'), ('2012', 'val')]  # 年份类别
classes, _ = get_classes(classes_path)

# 数量统计
photo_nums = np.zeros(len(VOCdevkit_sets))
nums = np.zeros(len(classes))


def convert_annotation(year, image_id, list_file):
    in_file = open(os.path.join(VOCdevkit_path, 'VOC%s\\Annotations\\%s.xml' % (year, image_id)), encoding='utf-8')
    tree = ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        different = 0
        if obj.find('different') != None:
            d = obj.find('different').text
        cls = obj.find('name').text
        if cls not in classes or int(different) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(float(xmlbox.find('xmin').text)), int(float(xmlbox.find('ymin').text)),
             int(float(xmlbox.find('xmax').text)), int(float(xmlbox.find('ymax').text)))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))

        nums[classes.index(cls)] = nums[classes.index(cls)] + 1


random.seed(0)

xml_file_path = os.path.join(VOCdevkit_path, 'VOC2012\\Annotations')
save_Base_Path = os.path.join(VOCdevkit_path, 'VOC2012\\ImageSets\\Main')
temp_xml = os.listdir(xml_file_path)  # Annotations中的标签
total_xml = []  # 储存标签
for xml in temp_xml:
    if xml.endswith(".xml"):
        total_xml.append(xml)

# 生成训练集和验证集
num = len(total_xml)  # 总数
list = range(num)
tv = int(num * trainval_percent)
tr = int(tv * train_percent)
trainval = random.sample(list, tv)
train = random.sample(trainval, tr)

# 储存训练集和验证集
ftrainval = open(os.path.join(save_Base_Path, 'trainval.txt'), 'w')
ftest = open(os.path.join(save_Base_Path, 'test.txt'), 'w')
ftrain = open(os.path.join(save_Base_Path, 'train.txt'), 'w')
fval = open(os.path.join(save_Base_Path, 'val.txt'), 'w')

for i in list:
    name = total_xml[i][:-4] + '\n'
    if i in trainval:
        ftrainval.write(name)
        if i in train:
            ftrain.write(name)
        else:
            fval.write(name)
    else:
        ftest.write(name)

ftrainval.close()
ftrain.close()
fval.close()
ftest.close()
print("done")

type_index = 0
for year, image_set in VOCdevkit_sets:  # 2012 train/val
    image_ids = open(os.path.join(VOCdevkit_path, 'VOC%s\\ImageSets\\Main\\%s.txt' % (year, image_set)),
                     encoding='utf-8').read().strip().split()  # 去除尾符后切割
    list_file = open('%s_%s.txt' % (year, image_set), 'w', encoding='utf-8')
    for image_id in image_ids:
        list_file.write('%s\\VOC%s\\JPEGImages\\%s.jpg' % (os.path.abspath(VOCdevkit_path), year, image_id))
        convert_annotation(year, image_id, list_file)
        list_file.write('\n')
    photo_nums[type_index] = len(image_ids)
    type_index += 1
    list_file.close()
print('done')