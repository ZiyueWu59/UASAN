
import numpy as np

from .common import voc_ap, iou


def trajectory_overlap(gt_trajs, pred_traj):
    """
    Calculate overlap among trajectories
    :param gt_trajs:
    :param pred_traj:
    :param thresh_s:
    :return:
    """
    max_overlap = 0
    max_index = 0
    thresh_s = [0.5, 0.7, 0.9]
    for t, gt_traj in enumerate(gt_trajs):
        top1, top2, top3 = 0, 0, 0
        total = len(set(gt_traj.keys()) | set(pred_traj.keys()))
        for i, fid in enumerate(gt_traj):
            if fid not in pred_traj:
                continue
            sIoU = iou(gt_traj[fid], pred_traj[fid])
            if sIoU >= thresh_s[0]:
                top1 += 1
                if sIoU >= thresh_s[1]:
                    top2 += 1
                    if sIoU >= thresh_s[2]:
                        top3 += 1

        tIoU = (top1 + top2 + top3) * 1.0 / (3 * total)

        if tIoU > max_overlap:
            max_overlap = tIoU
            max_index = t

    return max_overlap, max_index


def evaluate(gt, pred, use_07_metric=True, thresh_t=0.5):
    """
    Evaluate the predictions
    """
    gt_classes = set()
    for tracks in gt.values():
        for traj in tracks:
            gt_classes.add(traj['category'])
    gt_class_num = len(gt_classes)

    result_class = dict()
    for vid, tracks in pred.items():
        for traj in tracks:
            if traj['category'] not in result_class:
                result_class[traj['category']] = [[vid, traj['score'], traj['trajectory']]]
            else:
                result_class[traj['category']].append([vid, traj['score'], traj['trajectory']])

    ap_class = dict()
    print('Computing average precision AP over {} classes...'.format(gt_class_num))
    for c in gt_classes:
        if c not in result_class: 
            ap_class[c] = 0.
            continue
        npos = 0
        class_recs = {}

        for vid in gt:
            #print(vid)
            gt_trajs = [trk['trajectory'] for trk in gt[vid] if trk['category'] == c]
            det = [False] * len(gt_trajs)
            npos += len(gt_trajs)
            class_recs[vid] = {'trajectories': gt_trajs, 'det': det}

        trajs = result_class[c]
        vids = [trj[0] for trj in trajs]
        scores = np.array([trj[1] for trj in trajs])
        trajectories = [trj[2] for trj in trajs]

        nd = len(vids)
        fp = np.zeros(nd)
        tp = np.zeros(nd)

        sorted_inds = np.argsort(-scores)
        sorted_vids = [vids[id] for id in sorted_inds]
        sorted_traj = [trajectories[id] for id in sorted_inds]

        for d in range(nd):
            R = class_recs[sorted_vids[d]]
            gt_trajs = R['trajectories']
            pred_traj = sorted_traj[d]
            max_overlap, max_index = trajectory_overlap(gt_trajs, pred_traj)

            if max_overlap >= thresh_t:
                if not R['det'][max_index]:
                    tp[d] = 1.
                    R['det'][max_index] = True
                else:
                    fp[d] = 1.
            else:
                fp[d] = 1.

        # compute precision recall
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)

        rec = tp / float(npos)
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = voc_ap(rec, prec, use_07_metric)

        ap_class[c] = ap
    
    # compute mean ap and print
    print('=' * 30)
    ap_class = sorted(ap_class.items(), key=lambda ap_class: ap_class[0])
    total_ap = 0.
    for i, (category, ap) in enumerate(ap_class):
        print('{:>2}{:>20}\t{:.4f}'.format(i+1, category, ap))
        total_ap += ap
    mean_ap = total_ap / gt_class_num 
    print('=' * 30)
    print('{:>22}\t{:.4f}'.format('mean AP', mean_ap))

    return mean_ap, ap_class


if __name__ == "__main__":
    """
    You can directly run this script from the parent directory, e.g.,
    python -m evaluation.video_object_detection val_object_groundtruth.json val_object_prediction.json
    """
    import json
    from argparse import ArgumentParser

    parser = ArgumentParser(description='Video object detection evaluation.')
    parser.add_argument('groundtruth', type=str,help='A ground truth JSON file generated by yourself')
    parser.add_argument('prediction', type=str, help='A prediction file')
    args = parser.parse_args()
    
    print('Loading ground truth from {}'.format(args.groundtruth))
    with open(args.groundtruth, 'r') as fp:
        gt = json.load(fp)
    print('Number of videos in ground truth: {}'.format(len(gt)))

    print('Loading prediction from {}'.format(args.prediction))
    with open(args.prediction, 'r') as fp:
        pred = json.load(fp)
    print('Number of videos in prediction: {}'.format(len(pred['results'])))

    mean_ap, ap_class = evaluate(gt, pred['results'])
