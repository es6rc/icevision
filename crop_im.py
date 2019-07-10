import os
from albu import read_tsv, filter_for_annotations
import shutil

DATA_ROOT = ""
IM_DIR = "im"
ANN_DIR = "ann"
AUG_ANN_DIR = "annotation-aug"

def get_maxbtm(data_root, annotation_dir):
    ann_files = filter_for_annotations(data_root, annotation_dir)
    btm_coords = []
    for idx, ann_filename in enumerate(ann_files):
        tsv = read_tsv(ann_filename)
        for index, row in tsv.iterrows():
            btm_coords.append([os.path.basename(ann_filename), row["ybr"]])
    coord_max = max(btm_coords, key=lambda x: x[-1])
    print(coord_max)
    return coord_max[-1]


def cropim(data_root, image_dir):
    pass


def checkbbox(data_root, ann_dir, aug_ann_dir):
    ann_files = filter_for_annotations(data_root, ann_dir)
    aug_ann_files = filter_for_annotations(data_root, aug_ann_dir)
    for aug_idx, aug_ann_filename in enumerate(aug_ann_files):
        aug_tsv = read_tsv(aug_ann_filename)
        org_ann_filename = os.path.join(data_root, ann_dir, os.path.basename(aug_ann_filename)[:-8]+'.tsv')
        org_tsv = read_tsv(org_ann_filename)

        # Double check all the aug_bbox tsv files
        comparison = aug_tsv.equals(org_tsv)  # We ommitted some classes
        same = True
        if not comparison:
            for indx, row1 in aug_tsv.iterrows():
                for idx, row2 in org_tsv.iterrows():  # Check by rows
                    if row1.equals(row2):
                        break
                    elif idx == org_tsv.shape[0]:
                        same = False
        assert same, "the augmentation_bbox generated wrong!"
    print("All bboxes generated correctly!")


def cp_org_ann_files(data_root, ann_dir, aug_dir, save_dir):
    if not os.path.exists(os.path.join(data_root, save_dir)):
        os.mkdir(os.path.join(data_root, save_dir))

    aug_ann_files = filter_for_annotations(data_root, aug_dir)
    for aug_idx, aug_ann_filename in enumerate(aug_ann_files):
        org_ann_filename = os.path.join(data_root, ann_dir, os.path.basename(aug_ann_filename)[:-8]+'.tsv')
        if not os.path.exists(org_ann_filename):
            break
        else:
            print(aug_idx, os.path.exists(org_ann_filename))
        dst_ann_filename = os.path.join(data_root, save_dir, os.path.basename(aug_ann_filename))
        shutil.copy2(org_ann_filename, dst_ann_filename)


# get_maxbtm(DATA_ROOT, ANN_DIR)
# checkbbox(DATA_ROOT, ANN_DIR, AUG_ANN_DIR)
SAVE_DIR = 'new_ann'
cp_org_ann_files(DATA_ROOT, ANN_DIR, AUG_ANN_DIR, SAVE_DIR)
