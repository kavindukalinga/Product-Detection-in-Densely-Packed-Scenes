import numpy as np
import os

# column indices in ground-truth annotations and detections CSV files
FILENAME_COL, X1_COL, Y1_COL, X2_COL, Y2_COL, CONFIDENCE_COL = range(6)


def read_boxes_file(path, has_confidence=False):
    """
    Read a file containing n boxes from m images
    :param path: path to file
    :param has_confidence: whether the file contains confidence information for detections
    :return: list of n file names, nX4 array of box coordinates [x1, y1, x2, y2], list of n confidence
    values (if has_confidence=True, or only zeros if has_confidence=False)
    """
    if not os.path.exists(path):
        raise Exception("Error: provided path does not exist {}".format(path))

    with open(path, 'r') as f:
        lines = f.readlines()
    split_lines = [x.strip().split(',') for x in lines]

    num_boxes = len(split_lines)
    filenames = []
    boxes = np.zeros([num_boxes, 4]).astype(float)
    confidence = np.zeros(num_boxes).astype(float)
    for i, data_str in enumerate(split_lines):
        filenames.append(data_str[FILENAME_COL])
        boxes[i, :] = float(data_str[X1_COL]), float(data_str[Y1_COL]), float(data_str[X2_COL]), float(data_str[Y2_COL])
        if has_confidence:
            confidence[i] = float(data_str[CONFIDENCE_COL])

    return filenames, boxes, confidence

