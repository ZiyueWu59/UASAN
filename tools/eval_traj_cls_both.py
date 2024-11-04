import root_path
import argparse
import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '5'
from tqdm import tqdm
from collections import defaultdict

import torch
from models.TrajClsModel import OpenVocTrajCls
import pickle

from dataloaders.dataset_vidvrd import VidVRDTrajDataset
from utils.utils_func import get_to_device_func,vIoU_broadcast
from utils.config_parser import parse_config_py
from utils.logger import LOGGER, add_log_to_file
from peft import PeftModel


def load_checkpoint(model,optimizer,scheduler,ckpt_path):
    checkpoint = torch.load(ckpt_path,map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optim_state_dict'])
    scheduler.load_state_dict(checkpoint['sched_state_dict'])
    crt_epoch = checkpoint["crt_epoch"]
    batch_size = checkpoint["batch_size"]

    return model,optimizer,scheduler,crt_epoch,batch_size



def eval_TrajClsOpenVoc_bsz1(dataset_class,model_class,args,topks=[5,10]):
    cfg_path =  args.cfg_path
    ckpt_path = args.ckpt_path
    output_dir=args.output_dir
    eval_split = args.eval_split
    save_tag = args.save_tag

    if output_dir is None:
        output_dir = os.path.dirname(cfg_path)
    log_dir = os.path.join(output_dir,'logfile/')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    log_path = os.path.join(log_dir,f'eval_{save_tag}.log')
    add_log_to_file(log_path)

    configs = parse_config_py(cfg_path)
    dataset_cfg = configs["eval_dataset_cfg"]
    model_cfg = configs["model_cfg"]
    eval_cfg = configs["eval_cfg"]
    device = torch.device("cuda")

    if eval_split is None:
        assert dataset_cfg["class_splits"] is not None
    else:
        if eval_split == "base":
            class_splits = ("base",)
        elif eval_split == "novel":
            class_splits = ("novel",)
        elif eval_split == "all":
            class_splits = ("base","novel")
        else:
            assert False
        dataset_cfg["class_splits"] = class_splits

    LOGGER.info("dataset config: {}".format(dataset_cfg))
    LOGGER.info("model config: {}".format(model_cfg))
    LOGGER.info("evaluate config: {}".format(eval_cfg))

    vIoU_th     = eval_cfg["vIoU_th"]

    model = model_class(model_cfg,is_train=False).to(device)
    
    model = PeftModel.from_pretrained(model, ckpt_path)

    model.eval()
    if hasattr(model,"reset_classifier_weights"):
        model.reset_classifier_weights(eval_split)


    LOGGER.info("preparing dataloader...")
    dataset = dataset_class(**dataset_cfg)

    collate_func = dataset.get_collator_func()
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        collate_fn = lambda x :x[0] ,
        num_workers = 0,
        drop_last= False,
        shuffle= False,
    )
    LOGGER.info("dataloader ready.")
    dataset_len = len(dataset)
    dataloader_len = len(dataloader)
    LOGGER.info(
        "len(dataset)=={},len(dataloader)=={}".format(dataset_len,dataloader_len)
    )

    LOGGER.info("start evaluating:")
    LOGGER.info("use config: {}".format(cfg_path))
    LOGGER.info("use device: {}; eval split: {}".format(device,eval_split))

    if isinstance(dataset,VidVRDTrajDataset):
        inference_for_vidvrd(model,device,dataloader,topks,vIoU_th)

    LOGGER.info(f"log saved at {log_path}")
    LOGGER.handlers.clear()

def inference_for_vidvrd(model,device,dataloader,topks,vIoU_th):

    total_gt = 0
    total_hit = 0
    total_hit_at_k = defaultdict(int)
    for data in tqdm(dataloader):

        (
            seg_tag,
            traj_infos,
            gt_annos,
            labels,
            batch_vit_feats
        ) = data   
        batch_vit_feats = batch_vit_feats.to(device)

        with torch.no_grad():

            cls_scores,cls_ids = model.forward_inference_bsz1(batch_vit_feats)

        cls_scores = cls_scores.cpu()
        cls_ids = cls_ids.cpu()
        cls_scores, argids  = torch.sort(cls_scores,dim=0,descending=True)
        for k in topks:
            argids_topk = argids[:k]
            n_det,n_hit_at_k,n_gt = eval_traj_recall_topK_per_seg(traj_infos,cls_ids,gt_annos,argids_topk,vIoU_th)
            total_hit_at_k[k] += n_hit_at_k
        total_gt += n_gt
    for k in topks:
        recall_at_k = total_hit_at_k[k] / total_gt
        LOGGER.info(f"total_hit_at_{k}={total_hit_at_k[k]},total_gt={total_gt},recall_at_{k}={recall_at_k}")
    

def eval_traj_recall_topK_per_seg(det_info,det_cls_ids,gt_anno,ids_topk,vIoU_th,traj_ids=None):
    
    if gt_anno is None:
        return 0,0,0

    det_trajs = det_info["bboxes"]  
    n_det = len(det_trajs)
    det_fstarts = det_info["fstarts"]  
    if traj_ids is not None:
        det_fstarts = det_fstarts[traj_ids]
        det_trajs = [det_trajs[idx] for idx in traj_ids.tolist()]

    gt_trajs = gt_anno["bboxes"]      
    gt_fstarts = gt_anno["fstarts"]  
    gt_labels = gt_anno["labels"]    
    n_gt = len(gt_labels)


    det_trajs = [det_trajs[idx] for idx in ids_topk.tolist()]

    det_fstarts = det_fstarts[ids_topk]
    det_cls_ids = det_cls_ids[ids_topk]

    viou_matrix = vIoU_broadcast(det_trajs,gt_trajs,det_fstarts,gt_fstarts)  

    cls_eq_mask = det_cls_ids[:,None] == gt_labels[None,:] 
    viou_matrix[~cls_eq_mask] = 0.0

    max_vious, gt_ids = torch.max(viou_matrix,dim=-1) 
    mask = max_vious > vIoU_th
    gt_ids[~mask] = -1
    hit_gt_ids = list(set(gt_ids.tolist()))  
    n_hit = (torch.as_tensor(hit_gt_ids) >=0).sum().item()

    return n_det,n_hit, n_gt


if __name__ == "__main__":

    import sys
    argv = ['python', \
        '--dataset_class',  'VidVRDTrajDataset', \
        '--model_class', 'OpenVocTrajCls', \
        '--cfg_path', 'experiments/TrajCls_VidVRD/traj_lora_cfg_.py', \
        '--ckpt_path', 'experiments/TrajCls_VidVRD/finetune_for_Ov_TrajCls', \
        '--eval_split', 'novel', \
        '--output_dir', 'experiments/TrajCls_VidVRD', \
        '--save_tag', 'finetune_for_Ov_TrajCls',
        ] 

    sys.argv = argv
    
    parser = argparse.ArgumentParser(description="Object Detection Demo")
    
    parser.add_argument("--model_class", type=str,help="...")
    parser.add_argument("--dataset_class", type=str,help="...")
    
    parser.add_argument("--cfg_path", type=str,help="...")
    parser.add_argument("--ckpt_path", type=str,help="...")
    parser.add_argument("--output_dir", type=str, help="...") 
    parser.add_argument("--save_tag", type=str,default="",help="...")
    parser.add_argument("--eval_split", type=str,default="novel",help="...")
     
    args = parser.parse_args()

    # for split in ["base","novel","all"]:
    model_class = eval(args.model_class)
    dataset_class = eval(args.dataset_class)

    eval_TrajClsOpenVoc_bsz1(
        dataset_class=dataset_class,
        model_class = model_class,
        args = args
    )
