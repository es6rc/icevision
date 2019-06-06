import albumentations as albu
import cv2
import numpy as np


def get_training_augmentation(min_area=0., min_visibility=0.):
    train_transform = [

        albu.OneOf([
            albu.MotionBlur(p=.2),
            albu.MedianBlur(blur_limit=3, p=0.1),
            albu.Blur(blur_limit=3, p=0.1),
        ], p=0.2),
        albu.OneOf([
            albu.CLAHE(clip_limit=2),
            albu.IAASharpen(),
            albu.IAAEmboss(),
        ], p=0.3),
        albu.OneOf([
            albu.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.15, p=0.1),
            albu.RandomShadow(p=0.1),
            albu.RandomBrightness(limit=0.3, p=0.2),
            albu.RandomRain(slant_lower=0, slant_upper=8, drop_length=0, blur_value=4,
                            brightness_coefficient=0.8, rain_type='heavy', p=0.1),
            albu.RandomSunFlare(p=0.2),
        ]),
        albu.OneOf([
            albu.RGBShift(p=0.1),
            albu.HueSaturationValue(p=0.3),
        ]),
        albu.OneOf([
            albu.HorizontalFlip(p=0.5),
            albu.RandomSizedCrop(min_max_height=(720, 1380), height=1380, width=720, interpolation=cv2.INTER_AREA)
        ], p=0.2)
    ]
    return albu.Compose(train_transform,
                        bbox_params={'format': 'coco', 'min_area': min_area, 'min_visibility': min_visibility,
                                     'label_fields': ['category_id']})


def get_validation_augmentation(min_area=0., min_visibility=0.):
    test_transform = [
        albu.PadIfNeeded(1024, 1024)
    ]
    return albu.Compose(test_transform,
                        bbox_params={'format': 'coco', 'min_area': min_area, 'min_visibility': min_visibility,
                                     'label_fields': ['category_id']})


def apply_augmentation(img, gt_bboxes, gt_labels, stage='train'):

    if stage == 'train':
        func = get_training_augmentation
    else:
        func = get_validation_augmentation

    annotations = {'image': img, 'bboxes': gt_bboxes, 'category_id': gt_labels}

    annotations = func()(**annotations)
    img = annotations['image'].copy()
    bboxes = list()
    gt_labels = list()
    for idx, bbox in enumerate(annotations['bboxes']):
        x_min, y_min, w, h = bbox
        x_min, x_max, y_min, y_max = int(x_min), int(x_min + w), int(y_min), int(y_min + h)
        gt_label = annotations['category_id'][idx]
        bboxes.append([x_min, y_min, x_max, y_max])
        gt_labels.append(gt_label)

    bboxes = np.array(bboxes, dtype=np.float32)
    gt_labels = np.array(gt_labels, dtype=np.int64)
    return img, bboxes, gt_labels
