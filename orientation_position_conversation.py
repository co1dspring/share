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

# 替换方向关系为更自然语言形式
def natural_relation(rel):
    mapping = {
        "left": "on the left side of",
        "right": "on the right side of",
        "above": "above",
        "below": "below",
        "in front of": "in front of",
        "behind": "behind"
    }
    return mapping.get(rel, rel)

# 所有可能的空间关系
all_relations = [
    "on the left side of", "on the right side of",
    "above", "below",
    "in front of", "behind"
]


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
    short_output = []
    cot_output = []
    for i, pair in enumerate(image_info['object_pairs']):
        obj1 = pair['obj1']
        obj2 = pair['obj2']
        # dist = pair['dist']
        image_path = pair['image_path']
        rels = get_relation(obj1, obj2, extrinsic)
        if rels == []:
            continue
        # print(rels)
        rel_phrase = ' and '.join([natural_relation(r) for r in rels])
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

        obj1_coor = [round(x,2) for x in get_coor_in_camera(obj1['3D_location'], extrinsic)]
        obj2_coor = [round(x,2) for x in get_coor_in_camera(obj2['3D_location'], extrinsic)]

        rotation_directions = ['facing the camera', 'facing away from the camera']
        for rotation_direction in rotation_directions:
            # 以obj1为位置，朝向相机或背向相机，观察obj2的相对方位
            u = (obj1['2D_bbox'][0]+obj1['2D_bbox'][2])/2
            v = (obj1['2D_bbox'][1]+obj1['2D_bbox'][3])/2
            question = f"If you are observing at the position of the {obj1_desc}, and {rotation_direction}, which side of you is the {obj2_desc} located?"

            # 构造答案
            if rotation_direction == 'facing the camera':
                if "left" in rels:
                    left_right_str = "on the left side of"
                else:
                    left_right_str = "on the right side of"
                if "in front of" in rels:
                    front_behind_str = "behind"
                else:
                    front_behind_str = "in front of"
            else:
                if "left" in rels:
                    left_right_str = "on the right side of"
                else:
                    left_right_str = "on the left side of"
                if "in front of" in rels:
                    front_behind_str = "in front of"
                else:
                    front_behind_str = "behind"
            answer = f"{obj2_desc} is {left_right_str} you and {front_behind_str} you"

            # 构造思维链
            cot = f"""if you are observing at the position of {obj1_desc} and turn {rotation_direction}, then from the perspective of camera, we should first locate {obj1_desc}, whose 3D location in camera coordinate system is {obj1_coor} meters. And the 3D location in camera coordinate system of target object {obj2_desc} is located in {obj2_coor} meters, by comparing the depth and computing the relative position, we can draw a conclusion that the {obj2_desc} is {left_right_str} you and {front_behind_str} you."""

            short_output.append({
                        # "image_id": image_info['image_name'],
                        "conversation": [
                            {"human": question, "assistant": answer}
                        ],
                        "images": [image_path],
                        "task_category": "middle_level(orientation_position)",
                        "cot": cot
                    })
            cot_output.append({
                # "image_id": image_info['image_name'],
                "conversation": [
                    {"human": question, "assistant": f"<think>{cot}</think><answer>{answer}</answer>"}
                ],
                "images": [image_path],
                "task_category": "middle_level(orientation_position)",
                # "cot": cot
            })

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
            intrinsic = image['intrinsic']
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
    task_name = 'orientation_position'
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
