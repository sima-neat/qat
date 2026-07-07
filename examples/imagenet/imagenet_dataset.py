#**************************************************************************
#||                        SiMa.ai CONFIDENTIAL                          ||
#||   Unpublished Copyright (c) 2024 SiMa.ai, All Rights Reserved.       ||
#**************************************************************************
# NOTICE:  All information contained herein is, and remains the property of
# SiMa.ai. The intellectual and technical concepts contained herein are
# proprietary to SiMa and may be covered by U.S. and Foreign Patents,
# patents in process, and are protected by trade secret or copyright law.
#
# Dissemination of this information or reproduction of this material is
# strictly forbidden unless prior written permission is obtained from
# SiMa.ai.  Access to the source code contained herein is hereby forbidden
# to anyone except current SiMa.ai employees, managers or contractors who
# have executed Confidentiality and Non-disclosure agreements explicitly
# covering such access.
#
# The copyright notice above does not evidence any actual or intended
# publication or disclosure  of  this source code, which includes information
# that is confidential and/or proprietary, and is a trade secret, of SiMa.ai.
#
# ANY REPRODUCTION, MODIFICATION, DISTRIBUTION, PUBLIC PERFORMANCE, OR PUBLIC
# DISPLAY OF OR THROUGH USE OF THIS SOURCE CODE WITHOUT THE EXPRESS WRITTEN
# CONSENT OF SiMa.ai IS STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE
# LAWS AND INTERNATIONAL TREATIES. THE RECEIPT OR POSSESSION OF THIS SOURCE
# CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS TO
# REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE, USE, OR
# SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.
#
#**************************************************************************
from collections import defaultdict
import logging


IMAGENETTE_WNID_TO_IMAGENET_INDEX = {
    "n01440764": 0,    # tench
    "n02102040": 217,  # English springer
    "n02979186": 482,  # cassette player
    "n03000684": 491,  # chain saw
    "n03028079": 497,  # church
    "n03394916": 566,  # French horn
    "n03417042": 569,  # garbage truck
    "n03425413": 571,  # gas pump
    "n03445777": 574,  # golf ball
    "n03888257": 701,  # parachute
}


def limit_samples_by_class(samples, samples_limit):
    if samples_limit is None or samples_limit >= len(samples):
        return samples

    by_class = defaultdict(list)
    for sample in samples:
        by_class[sample[1]].append(sample)

    limited_samples = []
    class_ids = sorted(by_class)
    sample_index = 0
    while len(limited_samples) < samples_limit:
        added = False
        for class_id in class_ids:
            class_samples = by_class[class_id]
            if sample_index < len(class_samples):
                limited_samples.append(class_samples[sample_index])
                added = True
                if len(limited_samples) == samples_limit:
                    break
        if not added:
            break
        sample_index += 1
    return limited_samples


def set_dataset_samples(dataset, samples):
    dataset.samples = samples
    dataset.imgs = samples
    dataset.targets = [sample[1] for sample in samples]


def target_transform_for_classes(classes):
    if classes and all(
        class_name in IMAGENETTE_WNID_TO_IMAGENET_INDEX for class_name in classes
    ):
        logging.info("Mapping Imagenette WNID labels to ImageNet-1K class indices.")
        return lambda target: IMAGENETTE_WNID_TO_IMAGENET_INDEX[classes[target]]
    return None


def apply_imagenet_target_transform(dataset):
    target_transform = target_transform_for_classes(dataset.classes)
    if target_transform is not None:
        dataset.target_transform = target_transform
    return dataset
