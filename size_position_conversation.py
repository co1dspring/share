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

def generate_volume_options(correct_value, num_options=3, delta_range=(0.01, 0.2)):
    options = []
    while len(options) < num_options - 1:
        delta = random.uniform(*delta_range)
        fake = correct_value + delta if random.random() > 0.5 else max(0.0001, correct_value - delta)
        fake = round(fake, 4)
        if abs(fake - correct_value) > 0.005 and fake not in options:
            options.append(fake)
    correct_value_rounded = round(correct_value, 4)
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

def calculate_3d_volume(lwh):
    return lwh[0]*lwh[1]*lwh[2]


def get_bbox_corners(location, size, rotation):
    # 1. 计算未旋转的局部角点
    half_size = np.array(size) / 2
    corners_local = np.array([
        [-1, -1, -1], [1, -1, -1], [-1, 1, -1], [1, 1, -1],
        [-1, -1, 1], [1, -1, 1], [-1, 1, 1], [1, 1, 1]
    ]) * half_size

    # 2. 处理旋转（假设rotation是四元数或旋转矩阵）
    if len(rotation) == 4:  # 四元数 [qx, qy, qz, qw]
        rot = Rotation.from_quat(rotation).as_matrix()
    elif len(rotation) == 9:  # 3x3旋转矩阵展平
        rot = np.array(rotation).reshape(3, 3)
    else:
        raise ValueError("Unsupported rotation format")

    # 3. 旋转角点
    corners_rotated = (rot @ corners_local.T).T

    # 4. 平移至世界坐标
    corners_world = corners_rotated + np.array(location)

    # 5. 计算最小/最大角点
    bbox_min = np.min(corners_world, axis=0)
    bbox_max = np.max(corners_world, axis=0)

    return bbox_min, bbox_max
def natural_relation_direction(rel):
    return {
        "left": "to the left of",
        "right": "to the right of",
        "above": "above",
        "below": "below",
        "in front of": "in front of",
        "behind": "behind"
    }.get(rel, rel)
def get_mean_depth(location, extrinsic):
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

    return point_camera
def get_relation(obj1, obj2, extrinsic, xy_thresh=0.4, depth_thresh=0.4):
    rel = []
    # 先把物体的世界坐标系坐标变换为相机3D坐标系，z轴朝外，x垂直，y水平
    obj1_camera_xyz = get_mean_depth(obj1['3D_location'], extrinsic)
    obj2_camera_xyz = get_mean_depth(obj2['3D_location'], extrinsic)
    x1 = obj1_camera_xyz[1,0]
    y1 = obj1_camera_xyz[0,0]
    z1 = obj1_camera_xyz[2,0]
    x2 = obj2_camera_xyz[1,0]
    y2 = obj2_camera_xyz[0,0]
    z2 = obj2_camera_xyz[2,0]

    # 垂直方向
    if x1 + xy_thresh < x2:
        rel.append("above")
    elif x1 > x2 + xy_thresh:
        rel.append("below")

    # 水平方向
    if y1 + xy_thresh < y2:
        rel.append("left")
    elif y1 > y2 + xy_thresh:
        rel.append("right")

    # 前后方向（基于 mean_depth）
    if z1 + depth_thresh < z2:
        rel.append("in front of")
    elif z1 > z2 + depth_thresh:
        rel.append("behind")

    return rel
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
    point_camera = R @ location + t

    return [float(point_camera[1,0]), float(point_camera[0,0]), float(point_camera[2,0])]#垂直 水平 深度
def create_QA(image_info, extrinsic, maxq_perimage = 20):

    # 随机选取两个物体
    # valid_items = [item for item in image_info['objects'] if item['category'] not in unexpected_categories]
    short_output = []
    cot_output = []
    direction_map = ['left', 'right', 'above', 'below', 'behind', 'in front of']
    mode = ['smallest', 'biggest']



    for i, item in enumerate(image_info['objects']):
        label = item["category"]
        # dist = calculate_3d_distance(item['3D_location'], camera_pos)
        # volume = calculate_3d_volume(item['3D_size'])

        if 'image_w_bbox' in item.keys():
            image_path = item['image_w_bbox']
            color = item['bbox_color']
            desc = f'{label}(annotated by {color} bounding box)'
        else:
            desc = label
            image_path = image_info['image_path']

        item_coor = [round(x, 2) for x in
                     get_coor_in_camera(item['3D_location'], extrinsic)]
        # for q_type in mode:
        #     for direction in direction_map:
        q_type = random.choice(mode)
        direction = random.choice(direction_map)
        direction_text = natural_relation_direction(direction)
        flag = 0
        if q_type == 'smallest':
            min_volume = float('inf')
            # 遍历其他物体，找到符合空间关系的物体
            for other_obj in image_info['objects']:
                rels = get_relation(other_obj, item, extrinsic)#other_obj相较于item的关系
                if direction in rels and calculate_3d_volume(other_obj['3D_size']) < min_volume:
                    flag = 1
                    min_volume = calculate_3d_volume(other_obj['3D_size'])
                    target_volume = min_volume*1000
                    target_obj = other_obj
        else:
            max_volume = -1
            # 遍历其他物体，找到符合空间关系的物体
            for other_obj in image_info['objects']:
                rels = get_relation(other_obj, item, extrinsic)  # other_obj相较于item的关系
                if direction in rels and calculate_3d_volume(other_obj['3D_size']) > max_volume:
                    flag = 1
                    max_volume = calculate_3d_volume(other_obj['3D_size'])
                    target_volume = max_volume*1000
                    target_obj = other_obj
        if flag == 0:
            continue

        target_coor = [round(x, 2) for x in
                     get_coor_in_camera(target_obj['3D_location'], extrinsic)]
        q1 = f"What is the {q_type} object {direction_text} {desc}?"
        # a1 = f"the {target_obj['category']} whose volume is {round(target_volume, 2)} cubic decimeters"
        a1 = f"The {target_obj['category']}, whose coordinates in 3D camera coordinate system is {target_coor} meters"

        distractors = [x for x in image_info['objects'] if x != target_obj and x != item]
        if len(distractors)<2:
            continue
        distractors = random.sample(distractors, 2)
        distractors = [f"The {x['category']}, whose coordinates in 3D camera coordinate system is {[round(c, 2) for c in get_coor_in_camera(x['3D_location'], extrinsic)]} meters" for x in distractors]
        options = [a1] + distractors
        random.shuffle(options)
        option_labels = ['(a)', '(b)', '(c)']
        q2 = f"What is the {q_type} object {direction_text} {desc}?\nThe options are:\n"
        q2 += '\n'.join([f"{label}. {opt}" for label, opt in
                         zip(option_labels, options)])
        a2 = option_labels[options.index(a1)]

        cot = f"""To figure out the {q_type} object {direction_text} {desc}, we should know objects' locations and then compute their volumes. First, the 3d location of {desc} in camera coordinate system is {item_coor} meters. Among all objects {direction_text} {desc}, the {target_obj['category']} located in {target_coor} meters is the {q_type}, and its volume can be calculated by its size of 3D bounding box {[round(x,2) for x in target_obj['3D_size']]} meters, so its volume is {round(target_volume, 2)} cubic decimeters and is the {q_type}."""
        short_output.append({
            "conversation": [
                {"human": q1, "assistant": a1},
                {"human": q2, "assistant": a2}
            ],
            "images": [image_path],
            "cot": cot,
            "task_category": "middle_level(size_position)"
        })
        cot_output.append({
            "conversation": [
                {"human": q1,
                 "assistant": f"<think>{cot}</think><answer>{a1}</answer>"},
                {"human": q2,
                 "assistant": f"<think>{cot}</think><answer>{a2}</answer>"}
            ],
            "images": [image_path],
            # "cot": cot,
            "task_category": "middle_level(size_position)"
        })


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
    task_name = 'size_position'
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
