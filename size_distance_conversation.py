import numpy as np
import os, json, random, math
from tqdm import tqdm
from collections import Counter

def count_values(data_list, key):
    return Counter(item[key] for item in data_list if key in item)

unexpected_categories = ['roof', 'wall', 'floor', 'ceiling', 'curtain']

def save_to_json(data, filename):
    # 写入JSON文件
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    return filename

def read_json(json_dir):
    with open(json_dir, 'r', encoding='utf-8') as file:
        data = json.load(file)  # 返回字典或列表
    return data

def generate_options(correct_value, delta_range=(0.05, 0.5)):
    options = []
    while len(options) < 2:
        delta = random.uniform(*delta_range)
        fake = correct_value + delta if random.random() > 0.5 else max(0.01, correct_value - delta)
        fake = round(fake, 3)
        if abs(fake - correct_value) > 0.01 and fake not in options:
            options.append(fake)
    correct_rounded = round(correct_value, 3)
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
def remove_duplicates(data_list, key):
    seen = set()
    unique_list = []
    for item in data_list:
        value = tuple(item[key]) if isinstance(item[key], list) else item[key]
        if value not in seen:
            seen.add(value)
            unique_list.append(item)
    return unique_list
def calculate_3d_volume(lwh):
    return lwh[0]*lwh[1]*lwh[2]
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
    #首先获取video对应的所有图像下的2D物体，然后根据3D坐标，去除重复物体
    # video_path = video['video_path']
    # image_idx = video['img_idx']
    short_output = []
    cot_output = []
    # objects = []
    # for idx in image_idx:
    #     image_info = images_info[idx]
    #     objects.extend(image_info['objects'])
    # 去重
    # uniq_objects = remove_duplicates(objects, '3D_location')
    # 去掉特殊类别
    valid_objects = [obj for obj in image_info['objects'] if obj["category"] not in unexpected_categories]
    # 统计每个类别的数量
    # counts = count_values(uniq_objects, "category")
    # 构造问题和答案
    for main_obj in valid_objects:

        main_label = main_obj["category"]
        main_center = main_obj["3D_location"]
        main_volume = calculate_3d_volume(main_obj['3D_size'])
        main_coor = [round(x, 2) for x in
                     get_coor_in_camera(main_obj['3D_location'], extrinsic)]

        if 'image_w_bbox' in main_obj.keys():
            image_path = main_obj['image_w_bbox']
            color = main_obj['bbox_color']
            desc = f'{main_label}(annotated by {color} bounding box)'
        else:
            desc = main_label
            image_path = image_info['image_path']

        max_distance = -1
        max_label = None
        max_volume = 0
        min_distance = float('inf')
        min_label = None
        min_volume = 0

        for other_obj in valid_objects:
            if other_obj != main_obj:
                other_center = other_obj["3D_location"]
                other_volume = calculate_3d_volume(other_obj['3D_size'])
                distance = calculate_3d_distance(main_center, other_center)

                if distance > max_distance:
                    max_distance = distance
                    max_location = other_obj['3D_location']
                    max_label = other_obj["category"]
                    max_volume = other_volume * 1000 #单位：立方分米
                    max_obj = other_obj
                if distance < min_distance:
                    min_distance = distance
                    min_location = other_obj['3D_location']
                    min_label = other_obj["category"]
                    min_volume = other_volume * 1000
                    min_obj = other_obj

        if max_label:
            # 简答题1
            short_question1 = f"What's the volume of the object farthest from {desc}?"
            short_answer1 = f'{round(max_volume,2)} cubic decimeter'
            # COT 1
            other_center = [round(x, 2) for x in
                     get_coor_in_camera(max_location, extrinsic)]
            main_center = [round(x, 2) for x in main_center]
            cot1 = f"""First, we should figure out which object is farthest from {desc}. It's easy to find that {desc} is located at {main_coor} meters in camera 3D coordinate system. In this scene, the {max_label} located at {other_center} meters is the farthest object from the {desc}, the distance from {max_label} to {desc} is {round(max_distance, 2)} meters. Then we can calculate the volume of the {max_label}, the length, width, and height of this object's 3D bounding box are {[round(x,2) for x in max_obj['3D_size']]} meters, and its volume is {round(max_volume, 2)} cubic decimeters."""
            # 构造输出格式
            short_output.append({
                "conversation": [
                    {
                        "human": short_question1,
                        "assistant": short_answer1
                    }
                ],
                "images": [
                    image_path
                ],
                "task_category": "middle_level(size_distance)",
                "cot": cot1
            })
            cot_output.append({
                "conversation": [
                    {
                        "human": short_question1,
                        "assistant": f"<think>{cot1}</think><answer>{short_answer1}</answer>"
                    }
                ],
                "images": [
                    image_path
                ],
                "task_category": "middle_level(size_distance)",
                # "cot": cot1
            })

        if min_label:
            # 简答题2
            short_question2 = f"what's the volume of the object closest to {desc}?"
            short_answer2 = f'{round(min_volume,2)} cubic decimeter'
            # COT 2
            other_center = [round(x, 2) for x in
                     get_coor_in_camera(min_location, extrinsic)]
            main_center = [round(x, 2) for x in main_center]
            cot2 = f"""First, we should figure out which object is farthest from {desc}. It's easy to find that {desc} is located at {main_coor} in camera 3D coordinate system. In this scene, the {min_label} located at {other_center} meters is the farthest object from the {desc}, the distance from {min_label} to {desc} is {round(min_distance, 2)} meters. Then we can calculate the volume of the {min_label}, the length, width, and height of this object's 3D bounding box are {[round(x,2) for x in min_obj['3D_size']]} meters, and its volume is {round(min_volume, 2)} cubic decimeters."""
            short_output.append({
                "conversation": [
                    {
                        "human": short_question2,
                        "assistant": short_answer2
                    }
                ],
                "images": [
                    image_path
                ],
                "task_category": "middle_level(size_distance)",
                "cot": cot2
            })
            cot_output.append({
                "conversation": [
                    {
                        "human": short_question2,
                        "assistant": f"<think>{cot2}</think><answer>{short_answer2}</answer>"
                    }
                ],
                "images": [
                    image_path
                ],
                "task_category": "middle_level(size_distance)",
                # "cot": cot2
            })


    return short_output, cot_output

def read_scene(scene, data_dir):
    scene_dir = os.path.join(data_dir, scene)
    image_titles = ['iphone', 'dslr']
    short_QA = []
    cot_QA = []
    for image_title in image_titles:
        json_file = os.path.join(scene_dir, f'{image_title}_new.json')
        json_data = read_json(json_file)
        images_data = json_data['images']
        videos_data = json_data['videos']
        # for video in videos_data:
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
    task_name = 'size_distance'
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
