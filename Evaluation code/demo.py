from utils import read_boxes_file
from eval import eval_multi_threshold


def main():
    # path to dataset, download from challenge website
    path_to_sku110k_dataset = '/home/user/SKU110K_fixed/'
    annotation_path = path_to_sku110k_dataset + '/annotations/annotations_test.csv'
    # path to example results, download from challenge website
    detection_results_path = 'example_results.csv'
    # detection_results_path = 'Goldman_etal_SKU110K_test_results.csv'

    avg_ap = eval_results(annotation_path, detection_results_path)
    print('AP at IoU=.50:.05:.95: {}'.format(avg_ap))


def eval_results(annotation_path, detection_results_path):
    """
    Evaluate detection results
    :param annotation_path: path to annotations file
    :param detection_results_path: path to detection results file
    :return: AP at IoU=.50:.05:.95
    """

    gt_filenames, gt_bboxes, _ = read_boxes_file(annotation_path, has_confidence=False)
    det_filenames, det_bboxes, det_confidences = read_boxes_file(detection_results_path, has_confidence=True)

    avg_ap = eval_multi_threshold(gt_filenames, gt_bboxes, det_filenames, det_bboxes, det_confidences)
    return avg_ap


if __name__ == '__main__':
    main()
