import os
from PIL import Image
import random
import xml.etree.ElementTree as ET

def split_dataset(dataset_dir, class_names, train_ratio=0.8):
    """
    将数据集划分为训练集和验证集，并重新组织文件结构。

    参数：
        dataset_dir(str)：原始数据集目录路径。
        class_names(list of str)：类别名称列表。
        train_ratio(float)：训练集比例，默认值为0.8。

    返回：
        无返回值。
    """
    # 构造图像文件名列表和标签文件名列表
    image_files = []
    label_files = []
    for filename in os.listdir(os.path.join(dataset_dir, "Annotations\\xml")):
        if not filename.endswith(".xml"):
            continue
        image_filename = os.path.splitext(filename)[0] + ".jpg"
        image_path = os.path.join(dataset_dir, "JPEGImages\\xml", image_filename)
        if not os.path.exists(image_path):
            # 尝试将 png 图片转换为 jpg 格式
            png_image_path = os.path.join(dataset_dir, "JPEGImages\\xml", os.path.splitext(filename)[0] + ".png")
            if os.path.exists(png_image_path):
                try:
                    with Image.open(png_image_path) as img:
                        img.save(image_path)
                except Exception as e:
                    print(f"Error when converting {png_image_path} to jpeg: {e}")
                    continue
            else:
                continue
        image_files.append(image_filename)
        label_files.append(filename)

    # 打乱文件名顺序
    data = list(zip(image_files, label_files))
    random.shuffle(data)
    image_files, label_files = zip(*data)

    # 计算训练集和验证集数量
    num_train = int(len(image_files) * train_ratio)
    num_valid = len(image_files) - num_train

    # 创建数据集目录结构
    os.makedirs(os.path.join(dataset_dir, "JPEGImages\\train"), exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, "JPEGImages\\valid"), exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, "Annotations\\train"), exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, "Annotations\\valid"), exist_ok=True)

    # 处理训练集
    for i in range(num_train):
        image_filename = image_files[i]
        label_filename = label_files[i]

        # 处理图像文件
        src_image_path = os.path.join(dataset_dir, "JPEGImages\\xml", image_filename)
        dst_image_path = os.path.join(dataset_dir, "JPEGImages\\train", image_filename)
        os.symlink(src_image_path, dst_image_path)

        # 处理标签文件
        src_label_path = os.path.join(dataset_dir, "Annotations\\xml", label_filename)
        dst_label_path = os.path.join(dataset_dir, "Annotations\\train", os.path.splitext(image_filename)[0] + ".txt")
        tree = ET.parse(src_label_path)
        with open(dst_label_path, "w") as f:
            for obj in tree.findall("object"):
                name = obj.find("name").text
                if name not in class_names:
                    continue
                cls_id = class_names.index(name)
                bndbox = obj.find("bndbox")
                width = int(tree.find("size").find("width").text)
                height = int(tree.find("size").find("height").text)
                xmin = float(bndbox.find("xmin").text) / width
                ymin = float(bndbox.find("ymin").text) / height
                xmax = float(bndbox.find("xmax").text) / width
                ymax = float(bndbox.find("ymax").text) / height
                x_center = (xmin + xmax) / 2.0
                y_center = (ymin + ymax) / 2.0
                bbox_width = xmax - xmin
                bbox_height = ymax - ymin
                f.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}\n")

    # 处理验证集
    for i in range(num_train, num_train + num_valid):
        image_filename = image_files[i]
        label_filename = label_files[i]

        # 处理图像文件
        src_image_path = os.path.join(dataset_dir, "JPEGImages\\xml", image_filename)
        dst_image_path = os.path.join(dataset_dir, "JPEGImages\\valid", image_filename)
        os.symlink(src_image_path, dst_image_path)

        # 处理标签文件
        src_label_path = os.path.join(dataset_dir, "Annotations\\xml", label_filename)
        dst_label_path = os.path.join(dataset_dir, "Annotations\\valid", os.path.splitext(image_filename)[0] + ".txt")
        tree = ET.parse(src_label_path)
        with open(dst_label_path, "w") as f:
            for obj in tree.findall("object"):
                name = obj.find("name").text
                if name not in class_names:
                    continue
                cls_id = class_names.index(name)
                bndbox = obj.find("bndbox")
                width = int(tree.find("size").find("width").text)
                height = int(tree.find("size").find("height").text)
                xmin = float(bndbox.find("xmin").text) / width
                ymin = float(bndbox.find("ymin").text) / height
                xmax = float(bndbox.find("xmax").text) / width
                ymax = float(bndbox.find("ymax").text) / height
                x_center = (xmin + xmax) / 2.0
                y_center = (ymin + ymax) / 2.0
                bbox_width = xmax - xmin
                bbox_height = ymax - ymin
                f.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}\n")


if __name__ == "__main__":
    # 类别名称列表
    class_names = ["Hero Red", "Hero Blue", "Hero"]

    # 划分训练集和验证集
    split_dataset("D:\\code\\yolov5-7.0\\labelImgXml", class_names, train_ratio=0.8)