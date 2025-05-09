import numpy as np
import os, json, random, math
from tqdm import tqdm

unexpected_categories = ['roof', 'wall', 'floor', 'ceiling']

def save_to_json(data, filename):
    # 写入JSON文件
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    return filename

def read_json(json_dir):
    with open(json_dir, 'r', encoding='utf-8') as file:
        data = json.load(file)  # 返回字典或列表
    return data

def generate_options(correct_value, delta_range=(10, 100)):
    options = []
    while len(options) < 2:
        delta = random.uniform(*delta_range)
        fake = correct_value + delta if random.random() > 0.5 else max(0.01, correct_value - delta)
        fake = round(fake, 2)
        if abs(fake - correct_value) > 0.01 and fake not in options:
            options.append(fake)
    correct_rounded = round(correct_value, 2)
    insert_index = random.randint(0, 2)
    options.insert(insert_index, correct_rounded)
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
    short_output = []
    cot_output = []
    for i, pair in enumerate(image_info['object_pairs']):
        obj1 = pair['obj1']
        obj2 = pair['obj2']
        dist = pair['dist']*100
        image_path = pair['image_path']
        if 'bbox_color' not in obj1.keys():
            obj1_desc = obj1['category']
        elif 'color' not in obj1.keys():
            obj1_desc = f"{obj1['category']}(annotated by {obj1['bbox_color']} bounding box)"
        else:
            obj1_desc = f"{obj1['category']}(annotated by {obj1['color']} bounding box)"
        if 'bbox_color' not in obj2.keys():
            obj2_desc = obj2['category']
        elif 'color' not in obj2.keys():
            obj2_desc = f"{obj2['category']}(annotated by {obj2['bbox_color']} bounding box)"
        else:
            obj2_desc = f"{obj2['category']}(annotated by {obj2['color']} bounding box)"
        obj1_coor = get_coor_in_camera(obj1['3D_location'], extrinsic)
        obj2_coor = get_coor_in_camera(obj2['3D_location'], extrinsic)


        # QA1
        q1 = f"What is the Euclidean distance between the {obj1_desc} and the {obj2_desc} in centimeters?"
        a1 = f"{round(dist,2)} centimeters"

        # QA2
        options, correct_idx = generate_options(dist)
        option_labels = ['(a)', '(b)', '(c)']
        options_str = "\n".join([f"{option_labels[i]}. {options[i]} centimeters" for i in range(3)])
        q2 = f"What is the Euclidean distance between the {obj1_desc} and the {obj2_desc} in centimeters?\nThe options are:\n{options_str}"
        a2 = option_labels[correct_idx]

        # 处理 bbox_3d_1 和 bbox_3d_2 里的 center 元素，保留两位小数
        bbox_3d_1_center = [round(x, 2) for x in obj1_coor]
        bbox_3d_2_center = [round(x, 2) for x in obj2_coor]

        cot = f"""First, to figure out the Euclidean distance between the {obj1_desc} and the {obj2_desc}, we should know their locations. The 3d location of {obj1_desc} in camera coordinate system is {bbox_3d_1_center} meters, the 3d location of {obj2_desc} in camera coordinate system is {bbox_3d_2_center} meters, so the distance between {obj1_desc} and {obj2_desc} is {round(dist,2)} centimeters."""
        short_output.append({
            # "image_id": image_info['image_name'],
            "conversation": [
                {"human": q1, "assistant": a1},
                {"human": q2, "assistant": a2}
            ],
            "images": [image_path],
            "task_category": "low_level(abs_distance)",
            "cot": cot
        })
        cot_output.append({
            # "image_id": image_info['image_name'],
            "conversation": [
                {"human": q1, "assistant": f"<think>{cot}</think><answer>{a1}</answer>"},
                {"human": q2, "assistant": f"<think>{cot}</think><answer>{a2}</answer>"}
            ],
            "images": [image_path],
            "task_category": "low_level(abs_distance)",
            # "cot": cot
        })
        if i > maxq_perimage:
            break

    return short_output, cot_output

def read_scene(scene, data_dir):
    scene_dir = os.path.join(data_dir, scene)
    image_titles = ['iphone', 'dslr']
    short_QA = []
    cot_QA = []
    for image_title in image_titles:
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
    task_name = 'abs_distance'
    data_dir = os.path.join(manu_dir, 'scannetpp_training_data')
    short_QA_json = os.path.join(short_QA_dir, f'{task_name}_qa_short.json')
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
