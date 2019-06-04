import albumentations as albu


def get_training_augmentation(min_area=0., min_visibility=0.):
    train_transform = [

        albu.HorizontalFlip(p=0.5)

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
