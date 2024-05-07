import os
import xml.etree.ElementTree as ET


def convert_xml_to_yolo(xml_file, output_dir):
    # 解析XML文件
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # 获取图片尺寸
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)

    # 初始化YOLO格式的txt文件内容
    yolo_format = []

    # 遍历所有的<object>标签
    for obj in root.iter('object'):
        # 提取边界框信息
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)

        # 计算YOLO格式的坐标（归一化到0-1之间）
        x_center = (xmin + xmax) / (2 * width)
        y_center = (ymin + ymax) / (2 * height)
        width_ratio = (xmax - xmin) / width
        height_ratio = (ymax - ymin) / height

        # YOLO格式：类别索引 x_center y_center width_ratio height_ratio
        # 这里假设类别索引为0（通常需要在代码外部或配置文件中指定）
        label = '140'  # 假设smoke的类别索引为0，根据实际情况修改
        yolo_line = f"{label} {x_center:.6f} {y_center:.6f} {width_ratio:.6f} {height_ratio:.6f}"
        yolo_format.append(yolo_line)

        # 写入YOLO格式的txt文件
    filename = os.path.splitext(os.path.basename(xml_file))[0] + '.txt'
    output_file = os.path.join(output_dir, filename)
    with open(output_file, 'w') as f:
        for line in yolo_format:
            f.write(line + '\n')

        # 遍历文件夹中的所有XML文件并转换


input_dir = '../dataset/吸烟/pp_smoke/Annotations'  # XML文件所在的文件夹路径
output_dir = '../dataset/吸烟/pp_smoke/images'  # 输出txt文件的文件夹路径
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for filename in os.listdir(input_dir):
    if filename.endswith('.xml'):
        xml_file = os.path.join(input_dir, filename)
        convert_xml_to_yolo(xml_file, output_dir)