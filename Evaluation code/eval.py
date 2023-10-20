import numpy as np
from collections import defaultdict

# adapted from https://github.com/rbgirshick/py-faster-rcnn


def eval_single_threshold(gt_imgs, gt_bboxes, det_imgs, det_bboxes, det_confidences, ovthresh=0.5):
    """
    Evaluate detection results using a single overlap threshold
    :param gt_imgs: list of n image ground-truth file names
    :param gt_bboxes: array of nX4 ground-truth bounding boxes
    :param det_imgs: list of m image detections' file names
    :param det_bboxes: array of mX4 detection bounding boxes
    :param det_confidences: m array of detection confidence scores
    :param ovthresh: overlap threshold to consider a detection as correct
    :return:
    """
    if not set(det_imgs).issubset(set(gt_imgs)):
        raise Exception('Error: detection results include images outside of the groundtruth annotations')

    num_positives = len(gt_imgs)

    # keep gt data separately for each image. The data includes the object bounding bboxes (bboxes) and an indication
    # of which objects were detected (detected)
    gt_img_indices = defaultdict(list)
    for i, gt_img in enumerate(gt_imgs):
        gt_img_indices[gt_img].append(i)
    gt_img_data = {img: {'bboxes': gt_bboxes[indices], 'detected': [False] * len(indices)}
                   for img, indices in gt_img_indices.items()}

    # sort detections by decreasing confidence
    sorted_ind = np.argsort(-det_confidences)
    sorted_scores = np.sort(-det_confidences)
    det_bboxes = det_bboxes[sorted_ind, :]
    det_imgs = [det_imgs[x] for x in sorted_ind]

    # iterate over detections and determine TPs and FPs
    tp = np.zeros(len(det_bboxes))
    fp = np.zeros(len(det_bboxes))
    jmax = -1
    for d, img_name in enumerate(det_imgs):
        bb = det_bboxes[d, :].astype(float)
        ovmax = -np.inf
        BBGT = gt_img_data[img_name]['bboxes'].astype(float)

        if BBGT.size > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                   (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                   (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > ovthresh:
            if not gt_img_data[img_name]['detected'][jmax]:
                tp[d] = 1.
                gt_img_data[img_name]['detected'][jmax] = True
            else:
                fp[d] = 1.
        else:
            fp[d] = 1.

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(num_positives)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec)
    return ap


def eval_multi_threshold(gt_imgs, gt_bboxes, det_imgs, det_bboxes, det_confidences, ovthresholds=None):
    """
    Evaluate detection results as the average results of multiple overlap thresholds
    :param gt_imgs: list of n image ground-truth file names
    :param gt_bboxes: array of nX4 ground-truth bounding boxes
    :param det_imgs: list of m image detections' file names
    :param det_bboxes: array of mX4 detection bounding boxes
    :param det_confidences: m array of detection confidence scores
    :param ovthresholds: list of overlap thresholds
    :return:
    """
    if ovthresholds is None:
        ovthresholds = np.arange(0.5, 1, 0.05)

    aps = []
    for threshold in ovthresholds:
        print('Evaluation results at threshold {}'.format(threshold))
        ap = eval_single_threshold(gt_imgs, gt_bboxes, det_imgs, det_bboxes, det_confidences, ovthresh=threshold)
        aps.append(ap)

    return np.mean(aps)


def voc_ap(rec, prec):
    """
    Compute AP
    :param rec: list of recall values at each detection
    :param prec: list of precision values at each detection
    :return:
    """
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap
