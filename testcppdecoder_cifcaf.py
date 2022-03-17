import cv2
import os
import openpifpaf
import numpy as np
from openpifpaf.plugins.coco.constants import (
    COCO_KEYPOINTS, COCO_PERSON_SIGMAS, COCO_UPRIGHT_POSE, COCO_PERSON_SCORE_WEIGHTS,
    COCO_PERSON_SKELETON,
)

import torch

# human_pose_15 and human_pose_15_new
COCO_KEYPOINTS_15 = [
    'nose',             # 1
    'neck',             # 2
    'right_shoulder',   # 3
    'right_elbow',      # 4
    'right_wrist',      # 5
    'left_shoulder',      # 6
    'left_elbow',       # 7
    'left_wrist',       # 8
    'right_hip',        # 9
    'right_knee',       # 10
    'right_ankle',      # 11
    'left_hip',         # 12
    'left_knee',        # 13
    'left_ankle',       # 14
    'centeroid'         # 15
]

# human_pose_15 and human_pose_15_new
COCO_PERSON_SIGMAS_15 = [  # s is the object scale in the OKS definition???
    0.055,  # nose
    0.073,  # neck          <==
    0.079,  # shoulders
    0.072,  # elbows
    0.062,  # wrists
    0.079,  # shoulders
    0.072,  # elbows
    0.062,  # wrists
    0.107,  # hips
    0.087,  # knees
    0.089,  # ankles
    0.107,  # hips
    0.087,  # knees
    0.089,  # ankles
    0.090,  # centroid      <==
]

# human_pose_15 and human_pose_15_new
COCO_UPRIGHT_POSE_15 = np.array([
    [0.0, 9.3, 2.0],  # 'nose',             # 1
    [0.0, 8.1, 2.0],  # 'neck',             # 2 <==
    [1.4, 8.0, 2.0],  # 'right_shoulder',   # 3
    [1.75, 6.2, 2.0],  # 'right_elbow',     # 4
    [1.75, 4.2, 2.0],  # 'right_wrist',     # 5
    [-1.4, 8.0, 2.0],  # 'left_shoulder',   # 6
    [-1.75, 6.0, 2.0],  # 'left_elbow',     # 7
    [-1.75, 4.0, 2.0],  # 'left_wrist',     # 8
    [1.26, 4.0, 2.0],  # 'right_hip',       # 9
    [1.4, 2.1, 2.0],  # 'right_knee',       # 10
    [1.4, 0.1, 2.0],  # 'right_ankle',      # 11
    [-1.26, 4.0, 2.0],  # 'left_hip',       # 12
    [-1.4, 2.0, 2.0],  # 'left_knee',       # 13
    [-1.4, 0.0, 2.0],  # 'left_ankle',      # 14
    [0.0, 6.0, 2.0],  # 'centeroid',        # 15 <==
])

# human_pose_15_new
COCO_PERSON_SKELETON_15_NEW = [
    (2, 1), (2, 3), (2, 6), (2, 15), (15, 3),
    (15, 6), (15, 9), (15, 12), (3, 4), (4, 5),
    (6, 7), (7, 8), (9, 10), (10, 11), (12, 13),
    (13, 14)
]

COCO_PERSON_SCORE_WEIGHTS_15 = [3.0] * 3 + [1.0] * (len(COCO_KEYPOINTS_15) - 3)

def get_custom_cifcaf():

    cif_bin = np.fromfile('cif_ds_test.bin', dtype=np.float32).reshape(15,5,49,85)
    caf_bin = np.fromfile('caf_ds_test.bin', dtype=np.float32).reshape(16,9,49,85)
    
    print("cif size:", cif_bin.shape)
    print("caf size:", caf_bin.shape)

    cif = openpifpaf.headmeta.Cif('cif', '15kp',
                                          keypoints=COCO_KEYPOINTS_15,
                                          sigmas=COCO_PERSON_SIGMAS_15,
                                          pose=COCO_UPRIGHT_POSE_15,
                                          draw_skeleton=COCO_PERSON_SKELETON_15_NEW,
                                          score_weights=COCO_PERSON_SCORE_WEIGHTS_15,
                                          )
    cif.head_index=0
    cif.base_stride=16
    cif.upsample_stride=2
    caf = openpifpaf.headmeta.Caf('caf', '15kp',
                                          keypoints=COCO_KEYPOINTS_15,
                                          sigmas=COCO_PERSON_SIGMAS_15,
                                          pose=COCO_UPRIGHT_POSE_15,
                                          skeleton=COCO_PERSON_SKELETON_15_NEW,
                                          )
    caf.head_index=1
    caf.base_stride=16
    caf.upsample_stride=2

    return cif_bin, cif, caf_bin, caf

def get_openpifpaf_cifcaf():
    cif_bin = np.load('cif_opp.bin.npy')
    caf_bin = np.load('caf_opp.bin.npy')

    print("cif size:", cif_bin.shape)
    print("caf size:", caf_bin.shape)

    cif = openpifpaf.headmeta.Cif('cif', 'cocokp',
                                          keypoints=COCO_KEYPOINTS,
                                          sigmas=COCO_PERSON_SIGMAS,
                                          pose=COCO_UPRIGHT_POSE,
                                          draw_skeleton=COCO_PERSON_SKELETON,
                                          score_weights=COCO_PERSON_SCORE_WEIGHTS)
    cif.head_index=0
    cif.base_stride=16
    cif.upsample_stride=2
    caf = openpifpaf.headmeta.Caf('caf', 'cocokp',
                                          keypoints=COCO_KEYPOINTS,
                                          sigmas=COCO_PERSON_SIGMAS,
                                          pose=COCO_UPRIGHT_POSE,
                                          skeleton=COCO_PERSON_SKELETON)
    caf.head_index=1
    caf.base_stride=16
    caf.upsample_stride=2

    return cif_bin, cif, caf_bin, caf

def cpp_decode(cif_bin, cif, caf_bin, caf):
    cif_metas = [cif]
    caf_metas = [caf]

    cpp_decoder = torch.classes.openpifpaf_decoder.CifCaf(
                len(cif_metas[0].keypoints),
                torch.LongTensor(caf_metas[0].skeleton) - 1,
            )

    initial_annotations_t = None
    initial_ids_t = None

    fields = [ torch.tensor(cif_bin), torch.tensor(caf_bin) ]

    print("cif metas: ", cif_metas)
    print("caf metas: ", caf_metas)

    print("fields cif: ", fields[cif_metas[0].head_index].shape)
    print("cif_metas stride: ", cif_metas[0].stride)
    print("fields caf: ", fields[caf_metas[0].head_index].shape)
    print("caf_metas stride: ", caf_metas[0].stride)

    annotations, annotation_ids = cpp_decoder.call(
                fields[cif_metas[0].head_index],
                cif_metas[0].stride,
                fields[caf_metas[0].head_index],
                caf_metas[0].stride,
            )

    annotations_py = []
    for ann_data, ann_id in zip(annotations, annotation_ids):
        ann = openpifpaf.annotation.Annotation(cif_metas[0].keypoints,
                            caf_metas[0].skeleton,
                            score_weights=cif_metas[0].score_weights)
        ann.data[:, :2] = ann_data[:, 1:3]
        ann.data[:, 2] = ann_data[:, 0]
        ann.joint_scales[:] = ann_data[:, 3]
        if ann_id != -1:
            ann.id_ = int(ann_id)
        annotations_py.append(ann)

    visualize_image = np.zeros([720, 1280, 3],dtype=np.uint8)
    annotation_painter = openpifpaf.show.AnnotationPainter()

    meta = {
        'dataset_index': 0, 
        'file_name': '', 
        'offset': [0., 0.], 'scale': [1., 1.], 
        'rotation': {'angle': 0.0, 'width': None, 'height': None}, 
        'valid_area': [   0.,    0., 1279., 719.], 
        'hflip': False, 'width_height': [1280, 720]
        }

    pred = [ann.inverse_transform(meta) for ann in annotations_py]

    with openpifpaf.show.image_canvas(visualize_image, 'out.jpg') as ax:
        annotation_painter.annotations(ax, pred)

    print("annotations: ", annotations, annotations.shape)


if __name__ == '__main__':
    # CIF field: [15, 5, 49, 85]
    # CAF field: [16, 9, 49, 85]
    cif_bin, cif, caf_bin, caf = get_custom_cifcaf()

    # CIF field: [17, 5, 151, 201
    # CAF field: [19, 8, 151, 201]
    # cif_bin, cif, caf_bin, caf = get_openpifpaf_cifcaf()

    cpp_decode(cif_bin, cif, caf_bin, caf)
