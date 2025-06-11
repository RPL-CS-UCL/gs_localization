import os
import numpy as np
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

def adjust_camera_params(camera_params, new_width, new_height):
    model, width, height, fx, fy, cx, cy = camera_params
    scale_x = new_width / width
    scale_y = new_height / height
    fx *= scale_x
    fy *= scale_y
    cx *= scale_x
    cy *= scale_y
    return (model, new_width, new_height, fx, fy, cx, cy)

def round_python3(number):
    rounded = round(number)
    if abs(number - rounded) == 0.5:
        return 2.0 * round(number / 2.0)
    return rounded

def pipeline(scene, base_path):
    llffhold = 5
    view_path = str('train') + '_views'
    os.chdir(base_path + scene)
    if os.path.exists(view_path):
        os.system('rm -r ' + view_path)
    os.mkdir(view_path)
    os.chdir(view_path)
    os.mkdir('created')
    os.mkdir('triangulated')
    os.mkdir('images')
    os.mkdir('train_images')
    os.system('colmap model_converter  --input_path ../sparse/0/ --output_path ../sparse/0/  --output_type TXT')

    images = {}
    with open('../sparse/0/images.txt', "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                image_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                camera_id = int(elems[8])
                image_name = elems[9]
                fid.readline().split()
                images[image_name] = elems[1:]

    img_list = sorted(images.keys(), key=lambda x: x)

    
    train_img_list = [c for idx, c in enumerate(img_list) if (idx % llffhold) == 3]
    test_img_list = [c for idx, c in enumerate(img_list) if (idx % llffhold) != 3]
    
    """
    test_size = 5  # Number of images needed for the test set

    # Uniformly sample 5 images from img_list as test_img_list
    test_img_list = np.linspace(0, len(img_list) - 1, test_size, dtype=int)
    test_img_list = [img_list[i] for i in test_img_list]

    # The remaining images are assigned to train_img_list
    train_img_list = [c for c in img_list if c not in test_img_list]
    """

    # test_img_list file path
    test_file_path = './triangulated/list_test.txt'
    # check whether exists, if yes then delete first
    if os.path.exists(test_file_path):
        os.remove(test_file_path)
    # write the test split info
    with open(test_file_path, "w") as f:
        for test_img_name in test_img_list:
            f.write(test_img_name + "\n")

    # train_img_list file path
    train_file_path = './triangulated/list_train.txt'
    # check whether exists, if yes then delete first
    if os.path.exists(train_file_path):
        os.remove(train_file_path)
    # write the train split info
    with open(train_file_path, "w") as f:
        for train_img_name in train_img_list:
            f.write(train_img_name + "\n")

    for img_name in img_list:
        os.system('cp ../images_4/' + img_name + '  images/' + img_name)

    for img_name in train_img_list:
        os.system('cp ../images_4/' + img_name + '  train_images/' + img_name)


    #os.system('cp ../sparse/0/cameras.txt created/.')

    # Sample one image to get the expected image width and height
    sample_img_path = os.path.join('../images_4', train_img_list[0])
    sample_img = cv2.imread(sample_img_path)
    new_height, new_width = sample_img.shape[:2]

    with open('../sparse/0/cameras.txt', 'r') as f:
        camera_lines = f.readlines()

    with open('created/cameras.txt', 'w') as f:
        for line in camera_lines:
            if line.startswith('#'):
                f.write(line)
                continue
            parts = line.strip().split()
            camera_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            fx = float(parts[4])
            fy = float(parts[5])
            cx = float(parts[6])
            cy = float(parts[7])

            new_model, new_width, new_height, new_fx, new_fy, new_cx, new_cy = adjust_camera_params(
                (model, width, height, fx, fy, cx, cy), new_width, new_height)

            new_line = f"{camera_id} {new_model} {new_width} {new_height} {new_fx} {new_fy} {new_cx} {new_cy}\n"
            f.write(new_line)

    with open('created/points3D.txt', "w") as fid:
        pass

    res = os.popen( 'colmap feature_extractor --database_path database.db --image_path train_images  --SiftExtraction.max_image_size 1600 --SiftExtraction.max_num_features 4096 --SiftExtraction.estimate_affine_shape 1 --SiftExtraction.domain_size_pooling 1').read()
    os.system( 'colmap exhaustive_matcher --database_path database.db --SiftMatching.guided_matching 1 --SiftMatching.max_num_matches 8192')
    db = COLMAPDatabase.connect('database.db')
    db_images = db.execute("SELECT * FROM images")
    img_rank = [db_image[1] for db_image in db_images]
    with open('created/images.txt', "w") as fid:
        for idx, img_name in enumerate(img_rank):
            data = [str(1 + idx)] + [' ' + item for item in images[os.path.basename(img_name)]] + ['\n\n']
            fid.writelines(data)

    os.system('colmap point_triangulator --database_path database.db --image_path train_images --input_path created  --output_path triangulated  --Mapper.ba_local_max_num_iterations 40 --Mapper.ba_local_max_refinements 3 --Mapper.ba_global_max_num_iterations 100')
    #os.system('colmap model_converter  --input_path triangulated --output_path triangulated  --output_type TXT')
    #os.system('colmap image_undistorter --image_path train_images --input_path triangulated --output_path dense')
    #os.system('colmap patch_match_stereo --workspace_path dense')
    #os.system('colmap stereo_fusion --workspace_path dense --output_path dense/fused.ply')
    # delte created file
    if os.path.exists('created'):
        shutil.rmtree('created')

    db.close()
    # delete database file
    if os.path.exists('database.db'):
        os.remove('database.db')

    # delete train_images file
    if os.path.exists('train_images'):
        shutil.rmtree('train_images')
    # delete images file
    if os.path.exists('images'):
        shutil.rmtree('images')

for scene in ['bicycle', 'bonsai', 'counter', 'garden',  'kitchen', 'room', 'stump', "flowers", "treehill"]:
    pipeline(scene, base_path = 'D:/gs-localization/datasets/360_v2/')  # please use absolute path!

