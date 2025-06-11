import os
import numpy as np
from pathlib import Path
import cv2
import sys
import sqlite3
import shutil


IS_PYTHON3 = sys.version_info[0] >= 3
MAX_IMAGE_ID = 2**31 - 1

CREATE_CAMERAS_TABLE = """CREATE TABLE IF NOT EXISTS cameras (
    camera_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    model INTEGER NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    params BLOB,
    prior_focal_length INTEGER NOT NULL)"""

CREATE_DESCRIPTORS_TABLE = """CREATE TABLE IF NOT EXISTS descriptors (
    image_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE)"""

CREATE_IMAGES_TABLE = """CREATE TABLE IF NOT EXISTS images (
    image_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    name TEXT NOT NULL UNIQUE,
    camera_id INTEGER NOT NULL,
    prior_qw REAL,
    prior_qx REAL,
    prior_qy REAL,
    prior_qz REAL,
    prior_tx REAL,
    prior_ty REAL,
    prior_tz REAL,
    CONSTRAINT image_id_check CHECK(image_id >= 0 and image_id < {}),
    FOREIGN KEY(camera_id) REFERENCES cameras(camera_id))
""".format(MAX_IMAGE_ID)

CREATE_TWO_VIEW_GEOMETRIES_TABLE = """
CREATE TABLE IF NOT EXISTS two_view_geometries (
    pair_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    config INTEGER NOT NULL,
    F BLOB,
    E BLOB,
    H BLOB,
    qvec BLOB,
    tvec BLOB)
"""

CREATE_KEYPOINTS_TABLE = """CREATE TABLE IF NOT EXISTS keypoints (
    image_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE)
"""

CREATE_MATCHES_TABLE = """CREATE TABLE IF NOT EXISTS matches (
    pair_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB)"""

CREATE_NAME_INDEX = \
    "CREATE UNIQUE INDEX IF NOT EXISTS index_name ON images(name)"

CREATE_ALL = "; ".join([
    CREATE_CAMERAS_TABLE,
    CREATE_IMAGES_TABLE,
    CREATE_KEYPOINTS_TABLE,
    CREATE_DESCRIPTORS_TABLE,
    CREATE_MATCHES_TABLE,
    CREATE_TWO_VIEW_GEOMETRIES_TABLE,
    CREATE_NAME_INDEX
])


def create_symlink(source_path, target_path):
    if os.path.exists(target_path):
        os.remove(target_path)  
    os.symlink(source_path, target_path)  

def array_to_blob(array):
    if IS_PYTHON3:
        return array.tostring()
    else:
        return np.getbuffer(array)

def blob_to_array(blob, dtype, shape=(-1,)):
    if IS_PYTHON3:
        return np.fromstring(blob, dtype=dtype).reshape(*shape)
    else:
        return np.frombuffer(blob, dtype=dtype).reshape(*shape)

class COLMAPDatabase(sqlite3.Connection):

    @staticmethod
    def connect(database_path):
        return sqlite3.connect(database_path, factory=COLMAPDatabase)

    def __init__(self, *args, **kwargs):
        super(COLMAPDatabase, self).__init__(*args, **kwargs)

        self.create_tables = lambda: self.executescript(CREATE_ALL)
        self.create_cameras_table = \
            lambda: self.executescript(CREATE_CAMERAS_TABLE)
        self.create_descriptors_table = \
            lambda: self.executescript(CREATE_DESCRIPTORS_TABLE)
        self.create_images_table = \
            lambda: self.executescript(CREATE_IMAGES_TABLE)
        self.create_two_view_geometries_table = \
            lambda: self.executescript(CREATE_TWO_VIEW_GEOMETRIES_TABLE)
        self.create_keypoints_table = \
            lambda: self.executescript(CREATE_KEYPOINTS_TABLE)
        self.create_matches_table = \
            lambda: self.executescript(CREATE_MATCHES_TABLE)
        self.create_name_index = lambda: self.executescript(CREATE_NAME_INDEX)

    def update_camera(self, model, width, height, params, camera_id):
        params = np.asarray(params, np.float64)
        cursor = self.execute(
            "UPDATE cameras SET model=?, width=?, height=?, params=?, prior_focal_length=1 WHERE camera_id=?",
            (model, width, height, array_to_blob(params),camera_id))
        return cursor.lastrowid

def round_python3(number):
    rounded = round(number)
    if abs(number - rounded) == 0.5:
        return 2.0 * round(number / 2.0)
    return rounded

def load_fewshot_images(file_path, current_scene):
    fewshot_images = set()
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith(current_scene):
                scene, seq, frame = line.split(' ')
                image_name = f"{seq}-{frame}-color.png"  # Modify this if your images have a different extension
                fewshot_images.add(image_name)
    return fewshot_images


def generate_full_file(scene, base_path):
    # Input file path
    train_file_path = os.path.join(base_path, scene, 'TrainSplit.txt')
    test_file_path = os.path.join(base_path, scene, 'TestSplit.txt')
    
    # Output file path
    output_train_file_path = os.path.join(base_path, scene, 'train_full.txt')
    output_test_file_path = os.path.join(base_path, scene, 'test_full.txt')
    
    # make the env clean
    if os.path.exists(output_train_file_path):
        os.remove(output_train_file_path)
    if os.path.exists(output_test_file_path):
        os.remove(output_test_file_path)

    # Train image output directory
    train_images_dir = os.path.join(base_path, scene, 'train_images_full')

    # If the training image directory exists, delete it and recreate it
    if os.path.exists(train_images_dir):
        shutil.rmtree(train_images_dir)
    os.mkdir(train_images_dir)

    images_dir = os.path.join(base_path, scene, 'images_full')
    if os.path.exists(images_dir):
        shutil.rmtree(images_dir)
    os.mkdir(images_dir)

    depths_dir = os.path.join(base_path, scene, 'depths_full')
    if os.path.exists(depths_dir):
        shutil.rmtree(depths_dir)
    os.mkdir(depths_dir)

    train_depths_dir = os.path.join(base_path, scene, 'train_depths_full')
    if os.path.exists(train_depths_dir):
        shutil.rmtree(train_depths_dir)
    os.mkdir(train_depths_dir)

    if not os.path.exists(train_file_path):
        print(f"Train file does not exist: {train_file_path}")
        return

    # Process all images in train_fewshot_all.txt as the training set
    with open(train_file_path, 'r') as infile, open(output_train_file_path, 'w') as train_outfile:
        for line in infile:
            sequence = line.strip()
            # Transform 'sequenceX' to 'seq-XX' 
            if sequence.startswith("sequence"):
                sequence_number = sequence[len("sequence"):]  # Extract the last character as the sequence number

                if len(sequence_number) > 1:
                    sequence_dir = f"seq-{sequence_number}"
                else:
                    sequence_dir = f"seq-0{sequence_number[-1]}"
            else:
                print(f"Warning: Unexpected sequence format: {sequence}")
                continue
            
            sequence_dir_path = os.path.join(base_path, scene, sequence_dir)
            
            if not os.path.exists(sequence_dir_path):
                print(f"Warning: Test sequence directory does not exist: {sequence_dir_path}")
                continue
            
            for image_name in os.listdir(sequence_dir_path):
                if image_name.endswith('color.png'):
                    formatted_name = f"{sequence_dir}-{image_name}"
                    formatted_name = formatted_name.replace(".color.png","-color.png")
                    train_outfile.write(f"{formatted_name}\n")

                    source_path = os.path.join(base_path, scene, sequence_dir, image_name)
                    target_path = os.path.join(train_images_dir, formatted_name) 
                    target_path2 = os.path.join(images_dir, formatted_name)

                    if os.path.exists(source_path):
                        create_symlink(source_path, target_path)
                        create_symlink(source_path, target_path2)

                    else:
                        print(f"Warning: {source_path} does not exist and cannot be copied.")

                    # Copy the corresponding depths to the depths_dir directory
                    depth_name_in_folder = f"{sequence_dir}/{image_name.split('.')[0]}.depth_cali.png"
                    source_path = os.path.join(base_path.replace("7scenes","7Scenes_additional"), scene, depth_name_in_folder)
                    target_path = os.path.join(depths_dir, formatted_name.replace("color","depth"))
                    target_path2 = os.path.join(train_depths_dir, formatted_name.replace("color","depth"))

                    if os.path.exists(source_path):
                        create_symlink(source_path, target_path)
                        create_symlink(source_path, target_path2)
                    else:
                        print(f"Warning: {source_path} does not exist and cannot be copied.")



    # Process the sequences specified in TrainSplit.txt and 
    # use all images from these sequences as the train set
    with open(test_file_path, 'r') as testfile, open(output_test_file_path, 'w') as test_outfile:
        for line in testfile:
            sequence = line.strip()
            # Transform 'sequenceX' to 'seq-XX' 
            if sequence.startswith("sequence"):
                sequence_number = sequence[len("sequence"):]  # Extract the last character as the sequence number

                if len(sequence_number) > 1:
                    sequence_dir = f"seq-{sequence_number}"
                else:
                    sequence_dir = f"seq-0{sequence_number[-1]}"
            else:
                print(f"Warning: Unexpected sequence format: {sequence}")
                continue
            
            sequence_dir_path = os.path.join(base_path, scene, sequence_dir)
            
            if not os.path.exists(sequence_dir_path):
                print(f"Warning: Test sequence directory does not exist: {sequence_dir_path}")
                continue
            
            for image_name in os.listdir(sequence_dir_path):
                if image_name.endswith('color.png'):
                    formatted_name = f"{sequence_dir}-{image_name}"
                    formatted_name = formatted_name.replace(".color.png","-color.png")
                    test_outfile.write(f"{formatted_name}\n")

                    source_path = os.path.join(base_path, scene, sequence_dir, image_name)
                    target_path = os.path.join(images_dir, formatted_name)
                
                    if os.path.exists(source_path):
                        create_symlink(source_path, target_path)

                    # Copy the corresponding depths to the depths_dir directory
                    depth_name_in_folder = f"{sequence_dir}/{image_name.split('.')[0]}.depth_cali.png"
                    source_path = os.path.join(base_path.replace("7scenes","7Scenes_additional"), scene, depth_name_in_folder)
                    target_path = os.path.join(depths_dir, formatted_name.replace("color","depth"))
                    if os.path.exists(source_path):
                        create_symlink(source_path, target_path)
                    else:
                        print(f"Warning: {source_path} does not exist and cannot be copied.")
                    
    print(f"Train file generated: {output_train_file_path}")
    print(f"Training images copied to: {train_images_dir}")
    print(f"Test file generated: {output_test_file_path}")


def pipeline(scene, base_path):
    os.chdir(base_path + scene)

    dir = 'sparse/0/'

    if os.path.exists(dir):
        shutil.rmtree(dir)  # Delete directory and all its contents
    os.makedirs(dir)  # Recreate the directory
    os.system(f'colmap model_converter --input_path D:/visloc_pseudo_gt_limitations/7scenes_reference_models/{scene}/sfm_gt --output_path sparse/0/ --output_type TXT')

    with open('sparse/0/images.txt', "r") as fid:
        lines = fid.readlines()

    modified_lines = []
    for line in lines:
        if line.startswith("#") or line.strip() == "":
            modified_lines.append(line)
            continue

        elems = line.split()
        if len(elems) == 10:
            image_name = elems[9]
            new_image_name = image_name.replace('/', '-')
            new_image_name = new_image_name.replace('.color.png', '-color.png')
            elems[9] = new_image_name
            modified_lines.append(" ".join(elems) + "\n")
        else:
            modified_lines.append(line)

    # Write the modified content back to ../sparse/0/images.txt
    with open('sparse/0/images.txt', "w") as fid:
        fid.writelines(modified_lines)


# Call the pipeline function to process the specified scene
for scene in ["chess", "fire", "heads", "office", "pumpkin", "redkitchen", "stairs"]:
#for scene in ["stairs"]:
    #pipeline(scene, base_path='D:/gs-localization/datasets/7scenes/')
    generate_full_file(scene, base_path='D:/gs-localization/datasets/7scenes/')
