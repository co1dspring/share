import numpy as np
import os, json, random, math, shutil, cv2
from tqdm import tqdm
from pathlib import Path

unexpected_categories = ['roof', 'wall', 'floor', 'ceiling']
colors_bgr = {
    "red": (0, 0, 255),
    "green": (0, 255, 0),
    "blue": (255, 0, 0),
    "yellow": (0, 255, 255),
    "cyan": (255, 255, 0),
    "magenta": (255, 0, 255),
    "white": (255, 255, 255),
    # "black": (0, 0, 0),
    "orange": (0, 165, 255),
    "purple": (128, 0, 128),
    "pink": (203, 192, 255),
    "gray": (128, 128, 128),
}
thickness = 5  # 线宽
def read_jsonl(file_path):
    """读取jsonl文件，返回包含所有JSON对象的列表"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data.append(json.loads(line.strip()))
            except json.JSONDecodeError as e:
                print(f"解析失败的行: {line.strip()}, 错误: {e}")
    return data
def save_to_json(data, filename):
    # 写入JSON文件
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    return filename

def read_json(json_dir):
    with open(json_dir, 'r', encoding='utf-8') as file:
        data = json.load(file)  # 返回字典或列表
    return data

def generate_options(correct_value, num_options=3, delta_range=(10, 100)):
    options = []
    while len(options) < num_options - 1:
        delta = random.uniform(*delta_range)
        if random.random() > 0.5:
            fake_value = correct_value + delta
        else:
            fake_value = max(0.01, correct_value - delta)
        fake_value = round(fake_value, 2)
        if abs(fake_value - correct_value) > 0.05 and fake_value not in options:
            options.append(fake_value)

    correct_value_rounded = round(correct_value, 2)
    insert_index = random.randint(0, 2)
    options.insert(insert_index, correct_value_rounded)
    return options, insert_index

def calculate_3d_distance(coord1, coord2):
    """
    计算两个三维坐标之间的欧几里得距离

    参数:
        coord1 (list/tuple): 第一个物体的三维坐标 [x1, y1, z1]
        coord2 (list/tuple): 第二个物体的三维坐标 [x2, y2, z2]

    返回:
        float: 两个坐标之间的距离
    """
    if len(coord1) != 3 or len(coord2) != 3:
        raise ValueError("坐标必须是三维的")

    dx = coord1[0] - coord2[0]  # x轴差值
    dy = coord1[1] - coord2[1]  # y轴差值
    dz = coord1[2] - coord2[2]  # z轴差值

    distance = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    return distance


def get_camera_position(extrinsic_matrix):
    """
    从4x4外参矩阵计算相机在世界坐标系中的坐标

    参数:
        extrinsic_matrix (np.ndarray): 4x4的外参矩阵

    返回:
        np.ndarray: 相机坐标 (X, Y, Z)
    """
    extrinsic_matrix = np.array(extrinsic_matrix)
    R = extrinsic_matrix[:3, :3]  # 提取旋转矩阵
    T = extrinsic_matrix[:3, 3]  # 提取平移向量
    camera_pos = -R.T @ T  # 计算相机位置
    return camera_pos.tolist()


def is_category_unique(item, valid_items):
    # 获取目标item的category
    target_category = item.get('category')

    # 统计valid_items中相同category的item数量
    count = 0
    for valid_item in valid_items:
        if valid_item.get('category') == target_category:
            count += 1
            if count > 1:
                return False  # 提前终止，优化性能

    return count == 1
def get_coor_in_camera(location, extrinsic):
    """
    计算世界坐标系中的点相对于相机的深度（相机坐标系下的Z值）。

    参数:
        location (list or np.ndarray): 世界坐标系中的3D点坐标 [x, y, z]。
        extrinsic (list or np.ndarray): 4x4的外参矩阵（世界坐标系到相机坐标系的变换矩阵）。
        intrinsic (list or np.ndarray): 3x3的内参矩阵（仅用于验证，实际深度计算不需要）。

    返回:
        float: 点在相机坐标系下的深度（Z值）。
    """
    # 将输入转换为NumPy数组
    location = np.array(location, dtype=np.float64).reshape(3, 1)
    extrinsic = np.array(extrinsic, dtype=np.float64)

    # 提取旋转矩阵R和平移向量t
    R = extrinsic[:3, :3]  # 3x3旋转矩阵
    t = extrinsic[:3, 3:]  # 3x1平移向量

    # 将世界坐标转换到相机坐标系
    # P_c = R @ P_w + t
    point_camera = R @ location + t

    # 深度是相机坐标系下的Z值
    # depth = point_camera[2, 0]

    return [float(point_camera[1,0]), float(point_camera[0,0]), float(point_camera[2,0])]#垂直 水平 深度
def create_QA(image_info, extrinsic, maxq_perimage = 20):

    # 随机选取两个物体
    valid_items = [item for item in image_info['objects'] if item['category'] not in unexpected_categories]
    short_output = []
    cot_output = []
    camera_pos = get_camera_position(extrinsic)
    # os.makedirs(target_dir, exist_ok=True)
    # original_image_path = f'{data_dir}/{image_info['image_path']}'
    # target_image_path = f'{target_dir}/{image_info['image_path']}'
    # image = cv2.imread(original_image_path)
    # print(original_image_path, target_image_path)
    # cv2.imwrite(target_image_path, image)
    # print(os.file.exists(original_image_path), original_image_path)
    # shutil.copy2(original_image_path, target_image_path)



    for i, item in enumerate(image_info['objects']):
        label = item["category"]
        dist = calculate_3d_distance(item['3D_location'], camera_pos)
        dist = round(dist*100, 2)
        item_coor = get_coor_in_camera(item['3D_location'], extrinsic)
        item_coor = [round(x, 2) for x in item_coor]

        if 'image_w_bbox' in item.keys():
            image_path = item['image_w_bbox']
            color = item['bbox_color']
            # 问题1和答案1
            q1 = f"How far away is the {label} in the {color} bounding box from the camera in centimeters?"
            a1 = f"{dist} centimeters"

            # 问题2和答案2（选择题）
            options, correct_index = generate_options(dist)
            option_labels = ['(a)', '(b)', '(c)']
            options_str = "\n".join(
                [f"{option_labels[i]}. {options[i]}" for i in range(3)])
            q2 = f"How far away is the {label} in the {color} bounding box from the camera in centimeters?\nThe options are:\n{options_str}"
            a2 = option_labels[correct_index]

            cot = f"The 3d location of {label}(annotated by {color} bounding box) in camera coordinate system is {item_coor} meters, then we can calculate the distance from {label}(annotated by {color} bounding box) to camera, which is {dist} centimeters."
        else:
            image_path = image_info['image_path']
            # 问题1和答案1
            q1 = f"How far away is the {label} from the camera in centimeters?"
            a1 = f"{dist} centimeters"

            # 问题2和答案2（选择题）
            options, correct_index = generate_options(dist)
            option_labels = ['(a)', '(b)', '(c)']
            options_str = "\n".join(
                [f"{option_labels[i]}. {options[i]}" for i in range(3)])
            q2 = f"How far away is the {label} from the camera in centimeters?\nThe options are:\n{options_str}"
            a2 = option_labels[correct_index]

            cot = f"The 3d location of {label} in camera coordinate system is {item_coor}, then we can calculate the distance from {label} to camera, which is {dist} centimeters."

        short_output.append({
            # "image_id": image_info['image_name'],
            "conversation": [
                {"human": q1, "assistant": a1},
                {"human": q2, "assistant": a2}
            ],
            "images": [image_path],
            "task_category": "low_level(distance)",
            "cot": cot
        })
        cot_output.append({
            # "image_id": image_info['image_name'],
            "conversation": [
                {"human": q1,
                 "assistant": f"<think>{cot}</think><answer>{a1}</answer>"},
                {"human": q2,
                 "assistant": f"<think>{cot}</think><answer>{a2}</answer>"}
            ],
            "images": [image_path],
            "task_category": "low_level(distance)",
            # "cot": cot
        })
        if i > maxq_perimage:
            break


    return short_output, cot_output

def read_scene(scene, data_dir):
    scene_dir = f'{data_dir}/{scene}'#os.path.join(data_dir, scene)
    image_titles = ['iphone', 'dslr']
    short_QA = []
    cot_QA = []
    for image_title in image_titles:
        # if image_title == 'iphone':
        #     json_file = os.path.join(scene_dir, 'obj_annotation.json')
        # else:
        #     json_file = os.path.join(scene_dir, 'obj_annotation_dslr.json')
            # image_dir = os.path.join(scene_dir, image_title)
        json_file = os.path.join(scene_dir, f'{image_title}_new.json')
        images_data = read_json(json_file)['images']
        for image in images_data:
            # scene_id = image['scene_id']
            # image_name = image['image_name']
            # image_path = image['image_path']
            extrinsic = image['extrinsic']
            # intrinsic = image['intrinsic']
            # objects = image['objects']
            # for object in objects:
            #     category = object['category']
            #     location_3d = object['3D_location']
            #     size_3d = object['3D_size']
            #     rotation_3d = object['3D_rotation']
            #     bbox_2d = object['2D_bbox']
            # os.makedirs(os.path.join(target_dir, scene, image_title), exist_ok=True)
            short, cot = create_QA(image, extrinsic)
            short_QA.extend(short)
            cot_QA.extend(cot)
        return short_QA, cot_QA

if __name__ == "__main__":
    # manu_dir = 'D:/Data/scannetpp'
    manu_dir = './'
    short_QA_dir = os.path.join(manu_dir, 'scannetpp_short_QA')
    os.makedirs(short_QA_dir, exist_ok=True)
    cot_QA_dir = os.path.join(manu_dir, 'scannetpp_cot_QA')
    os.makedirs(cot_QA_dir, exist_ok=True)
    task_name = 'distance'
    data_dir = os.path.join(manu_dir, 'scannetpp_training_data')
    short_QA_json = os.path.join(short_QA_dir,
                                 f'{task_name}_qa_short.json')
    cot_QA_json = os.path.join(cot_QA_dir, f'{task_name}_qa_cot.json')
    scene_list = os.listdir(data_dir)
    scene_list = sorted(scene_list)
    short_QA = []
    cot_QA = []
    for scene in tqdm(scene_list):
        short, cot = read_scene(scene, data_dir)
        short_QA.extend(short)
        cot_QA.extend(cot)
    print(len(short_QA))
    save_to_json(short_QA, short_QA_json)
    save_to_json(cot_QA, cot_QA_json)
