
import root_path

import random
import numpy as np
import argparse
import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import json
from tqdm import tqdm
from collections import defaultdict, OrderedDict
import pickle
import torch
from torch.cuda.amp import autocast, GradScaler
from models.TrajClsModel import OpenVocTrajCls

from dataloaders.dataset_vidvrd import VidVRDUnifiedDataset_GIoU as VidVRDUnifiedDataset_GIoU_v1
from utils.config_parser import parse_config_py
from utils.utils_func import get_to_device_func
from utils.logger import LOGGER,add_log_to_file
from utils.evaluate import EvalFmtCvtor
from VidVRDhelperEvalAPIs import eval_visual_relation,evaluate_v2

from models.DistillModels_v1 import DistillModel_v13

from models.FinetuneModels_v1 import FinetuneQformerModels_v4

from peft import PeftModel

def get_merge_model(check_point):
    weight_dict = OrderedDict()
    k = 'base_model.model.'
    for key, value in check_point.items():
        if k in key:
            key = ''.join(k.split('k'))
        weight_dict[key] = value
    return weight_dict


def load_json(path):
    with open(path,'r') as f:
        x = json.load(f)
    return x

def eval_relation(
    model_class,
    dataset_class,
    args
    ):
    
    output_dir = args.output_dir
    save_tag = args.save_tag
    if args.output_dir is None:
        output_dir = os.path.dirname(args.cfg_path)
    log_dir = os.path.join(args.output_dir,'logfile/')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    log_path = os.path.join(log_dir,f'eval_{save_tag}.log')
    add_log_to_file(log_path)
    LOGGER.info("use args:{}".format(args))


    configs = parse_config_py(args.cfg_path)
    if args.eval_type == "SGDet":
        dataset_cfg = configs["eval_dataset_cfg"]
    elif args.eval_type == "PredCls" or args.eval_type == "SGCls":
        dataset_cfg = configs["GTeval_dataset_cfg"]
    
    configs["association_cfg"]["association_n_workers"] = args.asso_n_workers
    model_traj_cfg = configs["model_traj_cfg"]
    model_pred_cfg = configs["model_pred_cfg"]
    eval_cfg = configs["eval_cfg"]
    pred_topk = eval_cfg["pred_topk"]
    device = torch.device("cuda")
        

    LOGGER.info("preparing dataloader...")
    dataset = dataset_class(**dataset_cfg)

    LOGGER.info("dataset config: {}".format(dataset_cfg))
    LOGGER.info("model_traj config: {}".format(model_traj_cfg))
    LOGGER.info("model_pred config: {}".format(model_pred_cfg))
    LOGGER.info("evaluate config: {}".format(eval_cfg))

    model_traj = OpenVocTrajCls(model_traj_cfg,is_train=False)
    model_traj = PeftModel.from_pretrained(model_traj, 'experiments/TrajCls_VidVRD/finetune_for_Ov_TrajCls')

    model_traj = model_traj.to(device)
    model_traj.eval()
    model_traj.reset_classifier_weights(args.classifier_split_traj)


    model_pred = model_class(model_pred_cfg)
    LOGGER.info(f"loading check point from {args.ckpt_path_pred}")
    check_point = torch.load(args.ckpt_path_pred,map_location=torch.device("cpu"))
    if "model_state_dict" in check_point.keys():
        state_dict = check_point["model_state_dict"] 
    else:
        state_dict = check_point

    if hasattr(model_pred,"reset_classifier_weights"):
        if state_dict['classifier_weights'].size(1) == 61:
            split = 'novel'
        else:
            split = 'base'

        model_pred.reset_classifier_weights(split)  

    model_pred.load_state_dict(state_dict, strict=False)

    if hasattr(model_pred,"reset_classifier_weights"):
        model_pred.reset_classifier_weights(args.classifier_split_pred)
    model_pred.eval()
    model_pred = model_pred.to(device)
    to_device_func = get_to_device_func(device)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        collate_fn = lambda x : x[0] ,
        num_workers = 4,
        drop_last= False,
        shuffle= False,
    )
    LOGGER.info("dataloader ready.")
    dataset_len = len(dataset)
    dataloader_len = len(dataloader)
    LOGGER.info(
        "batch_size==1, len(dataset)=={} == len(dataloader)=={}".format(
            dataset_len,dataloader_len
        )
    )

    LOGGER.info("start evaluating:")
    LOGGER.info("use config: {}".format(args.cfg_path))
    LOGGER.info("eval_type: {}".format(args.eval_type))
    
    score_merge = "mul"
    convertor = EvalFmtCvtor(
        "vidvrd",
        args.enti_cls_split_info_path,
        args.pred_cls_split_info_path,
        score_merge=score_merge,
        segment_cvt=True
    )    
    infer_results_for_save = dict()
    
    count = 0

    for data in tqdm(dataloader):
  
        (
            seg_tag,
            det_traj_info,
            gt_traj_info,
            union_feats,
            rel_pos_feat,  
            gt_rel_pos_feaat,
            labels,
            gt_labels,
        ) = data

        
        traj_bboxes = det_traj_info["bboxes"]
        traj_starts = det_traj_info["fstarts"]
        traj_vit_feats = det_traj_info["vit_feats"]
        n_det = traj_vit_feats.shape[0]

        input_data = (            
            traj_vit_feats,
            union_feats,
            rel_pos_feat,)
        input_data = tuple(to_device_func(x) for x in input_data)
        vit_feats = to_device_func(det_traj_info['vit_feats'])

        with torch.no_grad():

            if args.eval_type == "SGDet" or args.eval_type == "SGCls":
                 traj_scores,traj_cls_ids = model_traj.forward_inference_bsz1(batch_vit_feats=vit_feats)
                
            elif  args.eval_type == "PredCls":
                traj_scores = torch.ones(size=(n_det,),device=device)
                traj_cls_ids = det_traj_info["cls_ids"].to(device)

            p_scores,p_clsids,pair_ids = model_pred.forward_inference_bsz1(input_data, args.classifier_split_pred, pred_topk)

        s_ids = pair_ids[:,0]  # (n_pair,)
        o_ids = pair_ids[:,1] 

        s_clsids = traj_cls_ids[s_ids]  # (n_pair,)
        o_clsids = traj_cls_ids[o_ids]
        s_scores = traj_scores[s_ids]
        o_scores = traj_scores[o_ids]

        n_pair,k = p_clsids.shape
        triplet_scores = torch.stack([
            p_scores.reshape(-1),
            s_scores[:,None].repeat(1,k).reshape(-1),
            o_scores[:,None].repeat(1,k).reshape(-1)
        ],dim=-1) # (n_pair*k,3)
        triplet_5tuple = torch.stack([
            p_clsids.reshape(-1),
            s_clsids[:,None].repeat(1,k).reshape(-1),
            o_clsids[:,None].repeat(1,k).reshape(-1),
            s_ids[:,None].repeat(1,k).reshape(-1),
            o_ids[:,None].repeat(1,k).reshape(-1),
        ],dim=-1) 

        infer_results_for_save[seg_tag] = {
            "traj_bboxes":[tb.cpu().clone() for tb in traj_bboxes],
            "traj_starts": traj_starts.cpu().clone(),
            "triplet_scores":triplet_scores.cpu().clone(),
            "triplet_5tuple":triplet_5tuple.cpu().clone(),
        }
    LOGGER.info("start to convert infer_results to json_format for eval ... score_merge=\'{}\'".format(score_merge))
    relation_results = dict()

    for seg_tag,results in tqdm(infer_results_for_save.items()):

        traj_bboxes = results["traj_bboxes"]
        traj_starts = results["traj_starts"]
        triplet_scores = results["triplet_scores"]
        triplet_5tuple = results["triplet_5tuple"]

        result_per_seg = convertor.to_eval_json_format(
            seg_tag,
            triplet_5tuple,
            triplet_scores,
            traj_bboxes,
            traj_starts,
            triplets_topk=eval_cfg["return_triplets_topk"],

        )
        relation_results.update(result_per_seg)

    if not args.segment_eval:
        LOGGER.info("start relation association ..., using config : {}".format(configs["association_cfg"]))
        relation_results = relation_association(configs["association_cfg"],relation_results)
    hit_infos = _eval_relation_detection_openvoc(
        args,
        prediction_results=relation_results,
        rt_hit_infos=True
    )

    save_path = os.path.join(output_dir,f"VidVRDtest_hit_infos_{save_tag}.pkl")
    LOGGER.info("save hit_infos to {}".format(save_path))
    with open(save_path,'wb') as f:
        pickle.dump(hit_infos,f)
    LOGGER.info("hit_infos saved.")
    
    if args.save_infer_results:
        save_path = os.path.join(output_dir,"infer_results_{}.pkl".format(save_tag))
        LOGGER.info("save infer_results to {}".format(save_path))
        with open(save_path,"wb") as f:
            pickle.dump(infer_results_for_save,f)
        LOGGER.info("results saved.")

    if args.save_json_results:
        save_path = os.path.join(output_dir,f"VidVRDtest_relation_results_{save_tag}.json")
        LOGGER.info("save results to {}".format(save_path))
        LOGGER.info("saving ...")
        with open(save_path,'w') as f:
            json.dump(relation_results,f)
        LOGGER.info("results saved.")
    
    LOGGER.info(f"log saved at {log_path}")
    LOGGER.handlers.clear()


def relation_association(config,segment_predictions):
    import multiprocessing
    from utils.association import parallel_association,greedy_graph_association,greedy_relation_association,nms_relation_association
    
    segment_tags = list(segment_predictions.keys())
    segment_prediction_groups = defaultdict(dict)
    for seg_tag in sorted(segment_tags):
        video_name, fstart, fend = seg_tag.split('-')  
        fstart,fend = int(fstart),int(fend)
        segment_prediction_groups[video_name][(fstart,fend)] = segment_predictions[seg_tag]
    video_name_list = sorted(list(segment_prediction_groups.keys()))

    print('start {} relation association using {} workers'.format(config['association_algorithm'], config['association_n_workers']))
    if config['association_algorithm'] == 'greedy':
        algorithm = greedy_relation_association
    elif config['association_algorithm'] == 'nms':
        algorithm = nms_relation_association
    elif config['association_algorithm'] == 'graph':
        algorithm = greedy_graph_association
    else:
        raise ValueError(config['association_algorithm'])


    video_relations = {}
    if config.get('association_n_workers', 0) > 0:
        with tqdm(total=len(video_name_list)) as pbar:
            pool = multiprocessing.Pool(processes=config['association_n_workers'])
            for vid in video_name_list:
                video_relations[vid] = pool.apply_async(parallel_association,
                        args=(vid, algorithm, segment_prediction_groups[vid], config),
                        callback=lambda _: pbar.update())
            pool.close()
            pool.join()
        for vid in video_relations.keys():
            res = video_relations[vid].get()
            video_relations[vid] = res
    else:
        for vid in tqdm(video_name_list):
            res = algorithm(segment_prediction_groups[vid], **config)
            video_relations[vid] = res

    return video_relations


def _eval_relation_detection_openvoc(
    args,
    prediction_results=None,
    rt_hit_infos = False,
    ):

    if prediction_results is None:
        LOGGER.info("loading json results from {}".format(args.json_results_path))
        prediction_results = load_json(args.json_results_path)
        LOGGER.info("Done.")
    else:
        assert args.json_results_path is None


    LOGGER.info("filter gt triplets with traj split: {}, predicate split: {}".format(args.target_split_traj,args.target_split_pred))
    traj_cls_info = load_json(args.enti_cls_split_info_path)
    pred_cls_info = load_json(args.pred_cls_split_info_path)
    traj_categories = [c for c,s in traj_cls_info["cls2split"].items() if (s == args.target_split_traj) or args.target_split_traj=="all"]
    traj_categories = set([c for c in traj_categories if c != "__background__"])
    pred_categories = [c for c,s in pred_cls_info["cls2split"].items() if (s == args.target_split_pred) or args.target_split_pred=="all"]
    pred_categories = set([c for c in pred_categories if c != "__background__"])

    if args.segment_eval:
        gt_relations = load_json(args.segment_gt_json)
    else:
        gt_relations = load_json(args.gt_json)

    gt_relations_ = defaultdict(list)
    for vsig,relations in gt_relations.items(): 
        for rel in relations:
            s,p,o = rel["triplet"]
            if not ((s in traj_categories) and (p in pred_categories) and (o in traj_categories)):
                continue
            gt_relations_[vsig].append(rel)
    gt_relations = gt_relations_
    if rt_hit_infos:
        mean_ap, rec_at_n, mprec_at_n,hit_infos, _ = evaluate_v2(gt_relations,prediction_results,viou_threshold=0.5)
    else:
        mean_ap, rec_at_n, mprec_at_n = eval_visual_relation(gt_relations,prediction_results,viou_threshold=0.5)
    LOGGER.info(f"mAP:{mean_ap}, Retection Recall:{rec_at_n}, Tagging Precision: {mprec_at_n}")
    LOGGER.info('detection mean AP (used in challenge): {}'.format(mean_ap))
    LOGGER.info('detection recall: {}'.format(rec_at_n))
    LOGGER.info('tagging precision: {}'.format(mprec_at_n))

    if rt_hit_infos:
        return hit_infos



if __name__ == "__main__":
    import sys
    random.seed(111)
    np.random.seed(111)
    torch.random.manual_seed(111)

    argv = [
        'python', \
        '--pred_cls_split_info_path', 'configs/VidVRD_pred_class_spilt_info_v2.json', \
        '--model_class', "DistillModel_v13", \
        '--dataset_class', 'VidVRDUnifiedDataset_GIoU_v1', 
        '--cfg_path', 'experiments/RelationCls_VidVRD/cfg_.py', \
        '--output_dir', 'experiments/RelationCls_VidVRD/', \
        '--target_split_traj', 'all', \
        '--target_split_pred', 'novel', \
        '--ckpt_path_pred', 'pre_trained_Ov_RelCls_model.pth', \
        '--save_tag', 'TaPn',
    ] 
    sys.argv = argv
    parser = argparse.ArgumentParser(description="Object Detection Demo")
    
    parser.add_argument("--cfg_path", type=str,help="...")
    parser.add_argument("--ckpt_path", type=str,help="...")
    parser.add_argument("--ckpt_path_traj", type=str,default="")
    parser.add_argument("--ckpt_path_pred", type=str,help="...")
    parser.add_argument("--enti_cls_split_info_path", type=str,default="configs/VidVRD_class_spilt_info.json")
    parser.add_argument("--pred_cls_split_info_path", type=str,default="configs/VidVRD_pred_class_spilt_info_v2.json")
    parser.add_argument("--output_dir", type=str, help="...")
    parser.add_argument("--model_class", type=str,help="...")
    parser.add_argument("--dataset_class", type=str,default="VidVRDUnifiedDataset")
    parser.add_argument("--segment_eval", action="store_true",default=False,help="...")
    parser.add_argument("--save_json_results", action="store_true",default=False,help="...")
    parser.add_argument("--save_infer_results", action="store_true",default=False,help="...")
    parser.add_argument("--target_split_traj", type=str,default="all",help="...")
    parser.add_argument("--target_split_pred", type=str,default="novel",help="...")    
    parser.add_argument("--save_tag", type=str,default="",help="...")
    parser.add_argument("--eval_type", type=str)
    parser.add_argument("--asso_n_workers", type=int,default=8)

    parser.add_argument("--gt_json", type=str,default="datasets/gt_jsons/VidVRDtest_gts.json",help="...")
    parser.add_argument("--segment_gt_json", type=str,default="datasets/gt_jsons/VidVRDtest_segment_gts.json",help="...")
    
     
    parser.add_argument("--json_results_path", type=str,help="...")
    args = parser.parse_args()
    

    args.classifier_split_traj = args.target_split_traj
    args.classifier_split_pred = args.target_split_pred


    dataset_class = eval(args.dataset_class)
    if args.model_class is not None:
        model_class = eval(args.model_class)
    
    
    if args.eval_type is None:
        args.save_tag = args.save_tag + "-".join(["PredCls","SGCls","SGDet"])

        for eval_type in [ "SGDet", "SGCls", "PredCls", ]:
            args.eval_type = eval_type
            eval_relation(model_class,dataset_class,args)

