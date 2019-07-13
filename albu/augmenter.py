import fnmatch
import json
import os
import re
import pandas as pd
import cv2
from .augmentations import get_training_augmentation, get_validation_augmentation
from converter import create_annotation_info

CATEGORIES = [
    {
        'id': 1,
        'name': '2.1',
        'supercategory': '2',
    },
    {
        'id': 2,
        'name': '2.4',
        'supercategory': '2',
    },
    {
        'id': 3,
        'name': '3.1',
        'supercategory': '3',
    },
    {
        'id': 4,
        'name': '3.24',
        'supercategory': '3',
    },
    {
        'id': 5,
        'name': '3.27',
        'supercategory': '3',
    },
    {
        'id': 6,
        'name': '4.1.1',
        'supercategory': '4.1',
    },
    {
        'id': 7,
        'name': '4.1.2',
        'supercategory': '4.1',
    },
    {
        'id': 8,
        'name': '4.1.3',
        'supercategory': '4.1',
    },
    {
        'id': 9,
        'name': '4.1.4',
        'supercategory': '4.1',
    },
    {
        'id': 10,
        'name': '4.1.5',
        'supercategory': '4.1',
    },
    {
        'id': 11,
        'name': '4.1.6',
        'supercategory': '4.1',
    },
    {
        'id': 12,
        'name': '4.2.1',
        'supercategory': '4.2',
    },
    {
        'id': 13,
        'name': '4.2.2',
        'supercategory': '4.2',
    },
    {
        'id': 14,
        'name': '4.2.3',
        'supercategory': '4.2',
    },
    {
        'id': 15,
        'name': '5.19.1',
        'supercategory': '5.19',
    },
    {
        'id': 16,
        'name': '5.19.2',
        'supercategory': '5.19',
    },
    {
        'id': 17,
        'name': '5.20',
        'supercategory': '5.20',
    },
    {
        'id': 18,
        'name': '8.22.1',
        'supercategory': '8.22',
    },
    {
        'id': 19,
        'name': '8.22.2',
        'supercategory': '8.22',
    },
    {
        'id': 20,
        'name': '8.22.3',
        'supercategory': '8.22',
    },
]


def read_tsv(file):
    df = pd.read_csv(file, sep="\t")
    return df


def filter_for_annotations(root, annotation_dir):
    file_types = ['*.tsv']
    file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
    file_name_prefix = '.*'  # does nothing here
    files = [os.path.join(root, annotation_dir, file) for file in os.listdir(os.path.join(root, annotation_dir))]
    files = [f for f in files if re.match(file_types, f)]
    files = [f for f in files if re.match(file_name_prefix, os.path.splitext(os.path.basename(f))[0])]

    return files


def save_augmented(annotations, coco_output, image_save_path, annotation_save_path):

    img = annotations['image'].copy()
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(image_save_path, img)

    tsv = list()
    for idx, bbox in enumerate(annotations['bboxes']):
        x_min, y_min, w, h = bbox
        x_min, x_max, y_min, y_max = int(x_min), int(x_min + w), int(y_min), int(y_min + h)

        occluded = bool(coco_output["annotations"][idx]['iscrowd'])
        temporary = coco_output["temporaries"][idx]

        tsv.append({
            "class": annotations['category_id'][idx],
            "xtl": x_min,
            "ytl": y_min,
            "xbr": x_max,
            "ybr": y_max,
            "temporary": temporary,
            "occluded": occluded,
            "data": "",
        })

    if len(tsv) > 0:
        df = pd.DataFrame(tsv)

        df.to_csv(annotation_save_path, sep="\t", index=False,
                  columns=['class', 'xtl', 'ytl', 'xbr', 'ybr', 'temporary', 'occluded', 'data'])


def orig2aug(categories, root,
             image_dir, annotation_dir,
             image_save_dir, annotation_save_dir,
             extension=".jpg", data="train"):

    im_save_path = os.path.join(root, image_save_dir)
    ann_save_path = os.path.join(root, annotation_save_dir)

    if not os.path.exists(im_save_path):
        os.mkdir(im_save_path)
    if not os.path.exists(ann_save_path):
        os.mkdir(ann_save_path)

    if categories is None:
        categories = CATEGORIES
    else:
        with open(categories, 'r') as file:
            categories = json.load(file)

    aug = get_training_augmentation() if data == "train" \
        else get_validation_augmentation()

    annotation_files = filter_for_annotations(root, annotation_dir)

    for idx, annotation_filename in enumerate(annotation_files):
        print('%dth of %d annotation files is being processed' %(idx, len(annotation_files)))
        tsv = read_tsv(annotation_filename)

        basename = os.path.basename(annotation_filename)
        image_name = os.path.splitext(basename)[0] + extension

        image_filename = os.path.join(root, image_dir, image_name)

        image = cv2.imread(image_filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_size = image.shape[0:2][::-1]

        coco_output = {
            "temporaries": [],
            "annotations": []
        }

        for index, row in tsv.iterrows():
            value = [x for x in categories if x["name"] == str(row["class"])]

            if value:
                value = value[0]
                class_id = value["id"]
                name = value["name"]
                category_info = {"id": class_id, "is_crowd": 0 if row["occluded"] is None else int(row["occluded"])}
                annotation_info = create_annotation_info(0, 0, category_info,
                                                         image_size, row.iloc[1:5])  # Dict type

                if annotation_info is not None:
                    annotation_info['name'] = name  # add "name" key to athe annotation_info
                    coco_output["annotations"].append(annotation_info)
                    coco_output["temporaries"].append(row["temporary"])

        annotations = {'image': image, 'bboxes': [d['bbox'] for d in coco_output['annotations']],
                       'category_id': [d['name'] for d in coco_output['annotations']]}

        if len(annotations['bboxes']) > 0:
            # if the image has bounding box data
            augmented = aug(**annotations)

            base_name = os.path.splitext(basename)[0] + '_aug'
            image_save_path = os.path.join(root, image_save_dir, base_name + extension)
            annotation_save_path = os.path.join(root, annotation_save_dir, base_name + ".tsv")

            # For each frame that it has
            save_augmented(augmented, coco_output, image_save_path, annotation_save_path)
