import os
import xml.etree.ElementTree as ET
from PIL import Image

def convert_txt_to_xml(txt_file_path, image_size, class_names, save_xml_path):
    # 读取 txt 文件中的物体信息
    with open(txt_file_path, 'r') as f:
        lines = f.readlines()

    # 创建 XML 根节点和子元素
    root = ET.Element("annotation")
    ET.SubElement(root, "folder").text = ""
    ET.SubElement(root, "filename").text = ""
    source_node = ET.SubElement(root, "source")
    ET.SubElement(source_node, "database").text = "Unknown"
    size_node = ET.SubElement(root, "size")
    ET.SubElement(size_node, "width").text = str(image_size[0])
    ET.SubElement(size_node, "height").text = str(image_size[1])
    ET.SubElement(size_node, "depth").text = "3"

    # 解析每个物体的信息并添加到 XML 中
    for line in lines:
        cls_idx, x_center, y_center, width, height = [float(x) for x in line.strip().split()]
        obj_node = ET.SubElement(root, "object")
        ET.SubElement(obj_node, "name").text = class_names[int(cls_idx)]
        bndbox_node = ET.SubElement(obj_node, "bndbox")
        xmin = int((x_center - width / 2) * image_size[0])
        ymin = int((y_center - height / 2) * image_size[1])
        xmax = int((x_center + width / 2) * image_size[0])
        ymax = int((y_center + height / 2) * image_size[1])
        ET.SubElement(bndbox_node, "xmin").text = str(xmin)
        ET.SubElement(bndbox_node, "ymin").text = str(ymin)
        ET.SubElement(bndbox_node, "xmax").text = str(xmax)
        ET.SubElement(bndbox_node, "ymax").text = str(ymax)

    # 将 XML 内容写入文件中
    tree = ET.ElementTree(root)
    tree.write(save_xml_path)


def convert_txt_files(txt_dir_path, class_names, save_xml_dir):
    # 遍历 txt 文件夹中的所有 txt 文件
    for txt_file_name in os.listdir(txt_dir_path):
        if not txt_file_name.endswith('.txt'):  # 如果不是 .txt 文件，则跳过
            continue
        txt_file_path = os.path.join(txt_dir_path, txt_file_name)

        # 构造保存的 XML 文件路径
        xml_file_name = os.path.splitext(txt_file_name)[0] + '.xml'
        xml_file_path = os.path.join(save_xml_dir, xml_file_name)

        # 获取对应图像文件的路径和尺寸
        image_file_path = None
        image_size = None
        for ext in ['.jpg', '.png']:
            temp_img_file_path = os.path.splitext(txt_file_path)[0] + ext
            temp_img_file_path = temp_img_file_path.replace('Annotations', 'JPEGImages')
            if os.path.isfile(temp_img_file_path):
                image_file_path = temp_img_file_path
                with Image.open(image_file_path) as im:
                    image_size = im.size
                break

        # 如果未找到对应的图像文件，则输出错误信息并跳过
        if image_file_path is None or image_size is None:
            print(f"无法找到 {os.path.basename(txt_file_path)} 对应的图像文件")
            continue

        # 调用 convert_txt_to_xml 函数进行转换
        convert_txt_to_xml(txt_file_path, image_size, class_names, xml_file_path)


if __name__ == "__main__":
    # 用法
    txt_dir_path = "D:\\code\\yolov5-7.0\\dataset\labels\\train"
    class_names = ["Hero Red", "Hero Blue", "Hero"]
    save_xml_dir = "labelImgXml"
    convert_txt_files(txt_dir_path, class_names, save_xml_dir)
