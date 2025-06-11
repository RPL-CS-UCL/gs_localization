import os
import sqlite3
import cv2
import numpy as np
import shutil

class COLMAPDatabase(sqlite3.Connection):

    @staticmethod
    def connect(database_path):
        return sqlite3.connect(database_path, factory=COLMAPDatabase)

def calculate_image_matches(database_path):
    conn = COLMAPDatabase.connect(database_path)
    cursor = conn.cursor()

    # 查询所有图像的名称
    cursor.execute("SELECT image_id, name FROM images")
    images = cursor.fetchall()

    # 创建一个字典来存储每个图像与其他图像的匹配数量
    match_counts = {image_id: 0 for image_id, _ in images}

    # 查询所有匹配对及其特征匹配数量
    cursor.execute("SELECT pair_id, rows FROM matches")
    matches = cursor.fetchall()

    for pair_id, num_matches in matches:
        image_id1 = pair_id >> 32
        image_id2 = pair_id & ((1 << 32) - 1)

        # 只在 image_id 存在于 match_counts 时才累加匹配数量
        if image_id1 in match_counts:
            match_counts[image_id1] += num_matches
        if image_id2 in match_counts:
            match_counts[image_id2] += num_matches

    conn.close()
    return match_counts, {image_id: name for image_id, name in images}


def find_low_match_images(match_counts, image_names, threshold):
    low_match_images = [(image_names[image_id], count) for image_id, count in match_counts.items() if count < threshold]

    print("以下图像的匹配点数量低于阈值：")
    for image_name, count in low_match_images:
        print(f"图像: {image_name}, 匹配点数量: {count}")

    return low_match_images

def pipeline(scene, base_path, match_threshold):
    view_path = 'views'
    os.chdir(base_path + scene)
    if os.path.exists(view_path):
        shutil.rmtree(view_path)
    os.mkdir(view_path)
    os.chdir(view_path)

    # Create folders for storing the results
    os.mkdir('sparse')
    os.mkdir('images')
    
    # Copy all images from the source directory to the images directory
    images_folder = os.path.join(base_path, scene, 'images')
    for img_file in os.listdir(images_folder):
        if img_file.endswith('.jpg'):
            shutil.copy(os.path.join(images_folder, img_file), 'images')
    
    # Perform feature extraction with specified camera model as PINHOLE
    os.system('colmap feature_extractor --database_path database.db --image_path images --ImageReader.camera_model PINHOLE --SiftExtraction.max_image_size 1600 --SiftExtraction.max_num_features 8192 --SiftExtraction.estimate_affine_shape 1 --SiftExtraction.domain_size_pooling 1')

    # Perform matching
    os.system('colmap exhaustive_matcher --database_path database.db --SiftMatching.guided_matching 1 --SiftMatching.max_num_matches 8192')

    # Calculate image matches and find low match images
    #match_counts, image_names = calculate_image_matches('database.db')
    #low_match_images = find_low_match_images(match_counts, image_names, match_threshold)

    # Optionally remove low match images (Uncomment if needed)
    # remove_low_match_images(match_counts, image_names, match_threshold, 'images')

    # Perform sparse reconstruction
    os.system('colmap mapper --database_path database.db --image_path images --output_path sparse')

    print("3D point cloud reconstruction completed with PINHOLE camera model.")

# Call the pipeline function to process the specified scene
pipeline(scene='forest', base_path='D:/gs-localization/datasets/', match_threshold=100)

