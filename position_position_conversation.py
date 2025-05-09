import numpy as np
import os, json, random, math
from tqdm import tqdm
from textwrap import dedent

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


def add_opposite_directions(input_list):
    """
    根据输入的方位词列表，随机添加其反义词，并确保：
    1. 输出不为空
    2. 如果必须随机添加词时，该词不在原输入列表中

    参数:
        input_list: 包含方位词的列表，如 ["left", "above"]

    返回:
        添加了可能反义词的新列表（至少包含一个词）
    """
    # 定义方位词及其反义词
    opposite_pairs = {
        "left": "right",
        "right": "left",
        "above": "below",
        "below": "above",
        "in front of": "behind",
        "behind": "in front of"
    }

    # 所有可能的方位词
    all_directions = list(opposite_pairs.keys())
    input_set = set(input_list)
    output = []

    # 首先尝试添加反义词（确保不重复）
    for word in input_list:
        if word in opposite_pairs:
            opposite = opposite_pairs[word]
            if opposite not in input_set and random.random() < 0.5:
                output.append(opposite)

    # 如果输出仍为空（输入为空或未添加任何词）
    if not output:
        # 从所有方位词中排除输入列表中的词
        available_directions = [d for d in all_directions if d not in input_set]
        # 如果所有词都在输入列表中（极端情况），则强制选择一个
        if not available_directions:
            available_directions = all_directions
        output.append(random.choice(available_directions))
    return output
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
        rels12 = get_relation(obj1, obj2, extrinsic)
        if rels12 == []:
            continue
        # print(rels)

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


        # 随机选择第三个物体，保证该物体在场景内唯一，这样不需要画新图像
        obj3 = None
        random.shuffle(image_info['objects'])
        for obj in image_info['objects']:
            if is_category_unique(obj, image_info['objects']):
                rels13 = get_relation(obj1, obj, extrinsic)
                if rels13 != []:
                    obj3 = obj
                    break
        if not obj3:
            continue
        obj3_desc = obj3['category']

        obj1_coor = [round(x, 2) for x in
                     get_coor_in_camera(obj1['3D_location'], extrinsic)]
        obj2_coor = [round(x, 2) for x in
                     get_coor_in_camera(obj2['3D_location'], extrinsic)]
        obj3_coor = [round(x, 2) for x in
                     get_coor_in_camera(obj3['3D_location'], extrinsic)]

        dir12_text = ' and '.join([natural_relation(r) for r in rels12])
        dir13_text = ' and '.join([natural_relation(r) for r in rels13])

        answers = ['yes', 'no']
        for answer in answers:
            correct_pos = random.choice(['a', 'b'])
            choices = {'a': "", 'b': ""}

            for key in choices:
                if key == correct_pos:
                    choices[key] = answer
                else:
                    choices[key] = [x for x in answers if x != answer][0]



            if answer == 'yes':
                question = f"Is the {obj1_desc} {dir12_text} the {obj2_desc} and {dir13_text} the {obj3_desc}?"
                question_mc = f"Is the {obj1_desc} {dir12_text} the {obj2_desc} and {dir13_text} the {obj3_desc}?The options are:\n(a). {choices['a']}\n(b). {choices['b']}"


            else:# 生成错误答案，保证和原答案不同
                wrong_rels12 = add_opposite_directions(rels12)
                wrong_rels13 = add_opposite_directions(rels13)
                wrong12_text = ' and '.join([natural_relation(r) for r in wrong_rels12])
                wrong13_text = ' and '.join([natural_relation(r) for r in wrong_rels13])
                question = f"Is the {obj1_desc} {wrong12_text} the {obj2_desc} and {wrong13_text} the {obj3_desc}?"
                question_mc = f"Is the {obj1_desc} {wrong12_text} the {obj2_desc} and {wrong13_text} the {obj3_desc}?The options are:\n(a). {choices['a']}\n(b). {choices['b']}"
            cot = f"""First, to figure out their spatial relationships, we should know their locations. The 3d location of {obj1_desc} in camera coordinate system is {obj1_coor} meters, the 3d location of {obj2_desc} in camera coordinate system is {obj2_coor} meters, and the 3d location of {obj3_desc} in camera coordinate system is {obj3_coor} meters. Then by computing their relative position in 3d space and comparing their depths, we can draw a conclusion that {obj1_desc} is {dir12_text} {obj2_desc} and {dir13_text} {obj3_desc}, so the answer is {answer}"""

        # short_output.append({
        #             # "image_id": image_info['image_name'],
        #             "conversation": [
        #                 {"human": question, "assistant": f"{obj1_desc} is {dir12_text} {obj2_desc} and {dir13_text} {obj3_desc}."},
        #                 {"human": question_mc, "assistant": f"({correct_pos})"}
        #             ],
        #             "images": [image_path],
        #             "task_category": "middle_level(position_position)",
        #             "cot": cot
        #         })
        short_output.append({
            # "image_id": image_info['image_name'],
            "conversation": [
                {"human": question, "assistant": answer},
                {"human": question_mc, "assistant": f"({correct_pos})"}
            ],
            "images": [image_path],
            "task_category": "middle_level(position_position)",
            "cot": cot
        })
        cot_output.append({
            # "image_id": image_info['image_name'],
            "conversation": [
                {"human": question,
                 "assistant": f"<think>{cot}</think><answer>{answer}</answer>"},
                {"human": question_mc, "assistant": f"<think>{cot}</think><answer>({correct_pos})</answer>"}
            ],
            "images": [image_path],
            "task_category": "middle_level(position_position)",
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
    task_name = 'position_position'
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
