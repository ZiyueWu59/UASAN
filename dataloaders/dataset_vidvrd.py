 
import os
import json
import pickle
from collections import defaultdict

from copy import deepcopy

import numpy as np
import torch
from tqdm import tqdm

from utils.logger import LOGGER
from utils.utils_func import vIoU_broadcast,vPoI_broadcast,trajid2pairid,bbox_GIoU, unique_with_idx_nd


def load_json(filename):
    with open(filename, "r") as f:
        x = json.load(f)
    return x


def prepare_segment_tags(dataset_dir,tracking_res_dir,dataset_splits):

    video_name_to_split = dict()
    for split in ["train","test"]:
        anno_dir = os.path.join(dataset_dir,split)
        for filename in sorted(os.listdir(anno_dir)):
            video_name = filename.split('.')[0]  
            video_name_to_split[video_name] = split
    
    segment_tags_all = [x.split('.')[0] for x in sorted(os.listdir(tracking_res_dir))] 
    segment_tags = []
    for seg_tag in segment_tags_all:
        video_name = seg_tag.split('-')[0] 
        split = video_name_to_split[video_name]
        if split in dataset_splits:
            segment_tags.append(seg_tag)
    
    return segment_tags


def _to_xywh(bboxes):
    x = (bboxes[...,0] + bboxes[...,2])/2
    y = (bboxes[...,1] + bboxes[...,3])/2
    w = bboxes[...,2] - bboxes[...,0]
    h = bboxes[...,3] - bboxes[...,1]
    return x,y,w,h


class VidVRDTrajDataset(object):

    def __init__(self,
        class_splits,
        dataset_split,
        class_spilt_info_path = "VidVRD-OpenVoc/configs/VidVRD_class_spilt_info.json",
        dataset_dir = "VidVRD_VidOR/vidvrd-dataset",
        tracking_res_dir = "VidVRD-II/tracklets_results/VidVRD_segment30_tracking_results_th-15-5",
        vit_path = "",
        cache_dir = "datasets/cache_vidvrd",
        vIoU_th = 0.5
    ):
        super().__init__()

        self.vIoU_th = vIoU_th
        self.vit_path = vit_path

        self.class_splits = tuple(cs.lower() for cs in class_splits)  
        self.dataset_split = dataset_split.lower()
        assert self.dataset_split in ("train","test")
        with open(class_spilt_info_path,'r') as f:
            self.class_split_info = json.load(f)

        self.dataset_dir = dataset_dir  
        self.tracking_res_dir = tracking_res_dir                  
        self.cache_dir = cache_dir
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        self.segment_tags = self.prepare_segment_tags()
        self.annotations = self.get_anno()

        LOGGER.info(" ---------------- dataset constructed len(self) == {} ----------------".format(len(self)))
    
    def __len__(self):

        return len(self.segment_tags)


    def prepare_segment_tags(self):

        print("preparing segment_tags for data_split: {}".format(self.dataset_split),end="... ")
        video_name_to_split = dict()
        for split in ["train","test"]:
            anno_dir = os.path.join(self.dataset_dir,split)
            for filename in sorted(os.listdir(anno_dir)):
                video_name = filename.split('.')[0] 
                video_name_to_split[video_name] = split
        
        segment_tags_all = [x.split('.')[0] for x in sorted(os.listdir(self.tracking_res_dir))] 
        segment_tags = []
        for seg_tag in segment_tags_all:
            video_name = seg_tag.split('-')[0] #
            split = video_name_to_split[video_name]
            if split == self.dataset_split:
                segment_tags.append(seg_tag)
        print("total: {}".format(len(segment_tags)))

        return segment_tags

    def get_traj_infos(self,seg_tag):

        path = os.path.join(self.tracking_res_dir,seg_tag + ".json")

        with open(path,'r') as f:
            tracking_results = json.load(f)

        fstarts = []
        scores = []
        bboxes = []
        cls_ids = []
        for ii,res in enumerate(tracking_results):

            
            fstarts.append(res["fstart"])  
            scores.append(res["score"])
            bboxes.append(torch.as_tensor(res["bboxes"])) 
            if 'label' in res.keys():
                cls_ids.append(res["label"])
            else:
                cls_ids = None

        traj_infos = {
            "fstarts":torch.as_tensor(fstarts), 
            "scores":torch.as_tensor(scores), 
            "bboxes":bboxes,  
            
        }
        if cls_ids is not None:
            traj_infos["VinVL_clsids"] = torch.as_tensor(cls_ids)
        
        return traj_infos


    def get_anno(self):
        
        print("preparing annotations for data_split: {}, class_splits: {} ".format(self.dataset_split,self.class_splits))

        annos = dict()
        anno_dir = os.path.join(self.dataset_dir,self.dataset_split)
        for filename in sorted(os.listdir(anno_dir)):
            video_name = filename.split('.')[0] 
            path = os.path.join(anno_dir,filename)

            with open(path,'r') as f:
                anno_per_video = json.load(f)  
            annos[video_name] = anno_per_video
        
        segment2anno_map = dict()
        for seg_tag in tqdm(self.segment_tags): 
            video_name, fstart, fend = seg_tag.split('-')  
            fstart,fend = int(fstart),int(fend)
            anno = annos[video_name]
            trajid2cls_map = {traj["tid"]:traj["category"] for traj in anno["subject/objects"]}

            trajs_info = {tid:defaultdict(list) for tid in trajid2cls_map.keys()}      
            annotated_len = len(anno["trajectories"])
            
            for frame_id in range(fstart,fend,1):
                if frame_id >= annotated_len:  
                    break

                frame_anno = anno["trajectories"][frame_id]  

                for bbox_anno in frame_anno:  
                    tid = bbox_anno["tid"]
                    bbox = bbox_anno["bbox"]
                    bbox = [bbox["xmin"],bbox["ymin"],bbox["xmax"],bbox["ymax"]]
                    trajs_info[tid]["bboxes"].append(bbox)
                    trajs_info[tid]["frame_ids"].append(frame_id)

            labels = []
            fstarts = []
            bboxes = []
            for tid, info in trajs_info.items():
                if not info:  
                    continue
                class_ = trajid2cls_map[tid]
                split_ = self.class_split_info["cls2split"][class_]
                if not (split_ in self.class_splits):
                    continue
                
                labels.append(
                    self.class_split_info["cls2id"][class_]
                )
                fstarts.append(
                    min(info["frame_ids"]) - fstart  
                )
                bboxes.append(
                    torch.as_tensor(info["bboxes"])  
                )

            if labels:
                labels = torch.as_tensor(labels)  
                fstarts = torch.as_tensor(fstarts)  
                traj_annos = {
                    "labels":labels,    
                    "fstarts":fstarts,  
                    "bboxes":bboxes,    
                }
            else:
                traj_annos = None
            
            segment2anno_map[seg_tag] = traj_annos
        
       
        return segment2anno_map
    
    
    def cls_label_assignment(self):
        print("constructing trajectory classification labels ... (vIoU_th={})".format(self.vIoU_th))
        
        assigned_labels = dict()

        for seg_tag in tqdm(self.segment_tags):
            det_info = self.get_traj_infos(seg_tag)
            det_trajs = det_info["bboxes"]   
            det_fstarts = det_info["fstarts"]  

            gt_anno = self.annotations[seg_tag]
            if gt_anno is None:
                assigned_labels[seg_tag] = None
                continue


            gt_trajs = gt_anno["bboxes"]     
            gt_fstarts = gt_anno["fstarts"]   
            gt_labels = gt_anno["labels"]     
            n_gt = len(gt_labels)

            viou_matrix = vIoU_broadcast(det_trajs,gt_trajs,det_fstarts,gt_fstarts) 

            max_vious, gt_ids = torch.max(viou_matrix,dim=-1) 
            mask = max_vious > self.vIoU_th
            gt_ids[~mask] = n_gt
            gt_labels_with_bg = torch.constant_pad_nd(gt_labels,pad=(0,1),value=0)
            assigned_labels[seg_tag] = gt_labels_with_bg[gt_ids]  
        
        return assigned_labels

    
    
    def __getitem__(self,idx):
        seg_tag = deepcopy(self.segment_tags[idx])   
        vit_feats = self.extract_features(seg_tag)
        if self.dataset_split == "train":
            traj_infos = None
            gt_annos = None
            labels = deepcopy(self.assigned_labels[seg_tag])
        else:
            traj_infos = self.get_traj_infos(seg_tag)
            gt_annos = deepcopy(self.annotations[seg_tag])
            labels = None

        return seg_tag, traj_infos, gt_annos, labels, vit_feats
    
    def extract_features(self, seg_tag):

        vit_path = os.path.join(self.vit_path,seg_tag+'.pkl')
        vit_feats = pickle.load(open(vit_path, 'rb'))
        vit_feats = torch.from_numpy(vit_feats.astype('float32'))        
        
        return vit_feats

        
    def get_collator_func(self):

        def collator_func(batch_data):
            bsz = len(batch_data)
            num_rt_vals = len(batch_data[0])
            return_values = tuple([batch_data[bid][vid] for bid in range(bsz)] for vid in range(num_rt_vals))
            return return_values

        return collator_func
    

class VidVRDUnifiedDataset(object):

    def __init__(self,
        dataset_splits,
        enti_cls_spilt_info_path = "configs/VidVRD_class_spilt_info.json",
        pred_cls_split_info_path = "", 
        dataset_dir = "VidVRD_VidOR/vidvrd-dataset",
        traj_info_dir = "VidVRD-II/tracklets_results/VidVRD_segment30_tracking_results", 
        vit_feat_dir = "",
        traj_union_dir = "",
        cache_dir = "VidVRD-OpenVoc/datasets/cache",
        gt_training_traj_supp = dict(
            traj_dir = "scene_graph_benchmark/output/VidVRD_tracking_results_gt",
            vit_feat_dir = "",
        ),
        pred_cls_splits = ("base",),
        traj_cls_splits = ("base",),
        traj_len_th = 15,
        min_region_th = 5,
        vpoi_th = 0.9,
        cache_tag = "",
        assign_label = None, 
        use_gt=True
        ):
        self.dataset_splits = tuple(ds.lower() for ds in dataset_splits) 
        self.traj_cls_splits = traj_cls_splits 
        self.pred_cls_splits  = pred_cls_splits
        if assign_label is None: 
            self.assign_label = False if "test" in self.dataset_splits else True
        else:
            self.assign_label = assign_label
        self.vit_feat_dir = vit_feat_dir
        self.traj_union_dir = traj_union_dir
        self.enti_cls_spilt_info = load_json(enti_cls_spilt_info_path)
        self.pred_cls_split_info = load_json(pred_cls_split_info_path)
        self.enti_cls2id = self.enti_cls_spilt_info["cls2id"] 
        self.pred_cls2id = self.pred_cls_split_info["cls2id"]
        self.enti_cls2split = self.enti_cls_spilt_info["cls2split"]
        self.pred_cls2split = self.pred_cls_split_info["cls2split"]
        self.pred_num_base = sum([v=="base" for v in self.pred_cls_split_info["cls2split"].values()])
        self.pred_num_novel = sum([v=="novel" for v in self.pred_cls_split_info["cls2split"].values()])

        self.dataset_dir = dataset_dir  
        self.traj_info_dir = traj_info_dir            
        self.cache_dir = cache_dir
        self.traj_len_th = traj_len_th
        self.min_region_th = min_region_th
        self.vpoi_th = vpoi_th
        self.gt_training_traj_supp = False
        self.cache_tag = cache_tag
        if not os.path.exists(self.cache_dir):
            os.makedirs(cache_dir)
        
        self.segment_tags = self.prepare_segment_tags() 
        self.video_annos = self.load_video_annos()

        if 'test' in self.dataset_splits:
            if 'gt' in self.vit_feat_dir:
                self.filter_results = None
            else:
                self.filter_results = pickle.load(open('filter_traj_test_results.pkl', 'rb')) # filter the trajs predicted as BackGround
            self.det_traj_infos = self.get_traj_infos()
        elif 'train' in self.dataset_splits:
            self.filter_results = None
            self.det_traj_infos = self.get_traj_infos()

        self.gt_traj_infos = None
        if gt_training_traj_supp is not None and use_gt:
            self.gt_training_traj_supp = True
            assert self.dataset_splits == ("train",)
            self.gt_traj_track_dir = gt_training_traj_supp["traj_dir"]
            self.gt_vit_feat_path = gt_training_traj_supp["vit_feat_dir"]
            self.merge_gt_traj()
        
        if self.dataset_splits == ("train",):

            segment_tags = self.filter_segments() 
            del_seg_tags = [seg_tag for seg_tag in segment_tags if self.det_traj_infos[seg_tag]['vit_feats'].size(0) <= 1]

            gt_vid_ap = pickle.load(open('train_gt_video_level_results_v1.pkl', 'rb'))['video_ap']
            filter_vid_tags = [k for k in gt_vid_ap if gt_vid_ap[k] < 0.3]
            seg_tag_map = json.load(open('seg_tag_map.json', 'r'))
            filter_list = [] 
            for k in filter_vid_tags:
                filter_list += seg_tag_map[k]
            segment_tags = sorted(list(set(segment_tags) - set(del_seg_tags)))
            segment_tags = sorted(list(set(segment_tags) - set(filter_list)))
            self.segment_tags = segment_tags 
            print("Delet segments with only on traj. {} segments left".format(len(self.segment_tags)))

        self.tag = 'wo_Bg_with_GT'
        if self.assign_label:
            assert self.dataset_splits == ("train",) , "we only do label assignment for train set"
            path_ = os.path.join(self.cache_dir, self.tag,"{}VidVRDtrain_Labels_th-{}-{}-{}.pkl".format(self.cache_tag,traj_len_th,min_region_th,vpoi_th))
            if os.path.exists(path_):
                print("assigned_labels loading from {}".format(path_))
                with open(path_,'rb') as f:
                    assigned_labels = pickle.load(f)
            else:
                print(f"no cache file found, assigning labels..., {path_}")
                assigned_labels = self.label_assignment()
                with open(path_,'wb') as f:
                    pickle.dump(assigned_labels,f)
                print("assigned_labels saved at {}".format(path_))
            print("len(assigned_labels) =",len(assigned_labels))
            segment_tags = sorted(assigned_labels.keys())  # 
            self.segment_tags = list(set(self.segment_tags) & set(segment_tags))
            self.assigned_labels = assigned_labels
        
        split_tag = "".join(self.dataset_splits)
        path_ = os.path.join(self.cache_dir, self.tag,"{}_VidVRD{}_rel_pos_features_th-{}-{}.pkl".format(self.cache_tag,split_tag,traj_len_th,self.min_region_th))
        if os.path.exists(path_):
            print("rel_pos_features loading from {}".format(path_))
            with open(path_,'rb') as f:
                rel_pos_features = pickle.load(f)
        else:
            print("no cache file found, ", end="")
            rel_pos_features = self.get_relative_position_feature()
            with open(path_,'wb') as f:
                pickle.dump(rel_pos_features,f)
            print("rel_pos_features saved at {}".format(path_))
        self.rel_pos_features = rel_pos_features
        if self.dataset_splits == ("train",):
            self.cache_tag += "OnlyWithGtTrainingData_"
            path_ = os.path.join(self.cache_dir, self.tag,"{}VidVRDtrain_Labels_th-{}-{}-{}.pkl".format(self.cache_tag,traj_len_th,min_region_th,vpoi_th))
            if os.path.exists(path_):
                print("assigned_labels loading from {}".format(path_))
                with open(path_,'rb') as f:
                    gt_labels = pickle.load(f)
                self.gt_labels = gt_labels
            else:
                print(f"no cache file found, assigning labels..., {path_}")
            path_ = os.path.join(self.cache_dir, self.tag,"{}VidVRD{}_rel_pos_features_for_gt_th-{}-{}.pkl".format(self.cache_tag,split_tag,traj_len_th,self.min_region_th))
            if not os.path.exists(path_):
                rel_pos_features = self.get_relative_position_feature(target='gt_traj')
                pickle.dump(rel_pos_features,open(path_, 'wb'))
                self.gt_rel_pos_feats = rel_pos_features

            else:    
                self.gt_rel_pos_feats = pickle.load(open(path_, 'rb'))
        else:
            self.gt_rel_pos_feats = None

        print("--------------- dataset constructed ---------------")


    def prepare_segment_tags(self):

        print("preparing segment_tags for data_splits: {}".format(self.dataset_splits),end="... ")
        video_name_to_split = dict()
        for split in ["train","test"]:
            anno_dir = os.path.join(self.dataset_dir,split)
            for filename in sorted(os.listdir(anno_dir)):
                video_name = filename.split('.')[0] 
                video_name_to_split[video_name] = split
        
        segment_tags_all = [x.split('.')[0] for x in sorted(os.listdir(self.traj_info_dir))] 
        segment_tags = []
        for seg_tag in segment_tags_all:
            video_name = seg_tag.split('-')[0]
            split = video_name_to_split[video_name]
            if split in self.dataset_splits:
                segment_tags.append(seg_tag)
        print("total: {}".format(len(segment_tags)))

        return segment_tags

    def is_filter_out(self,h,w,traj_len):
        if traj_len < self.traj_len_th:
            return True
        
        if  h < self.min_region_th or w < self.min_region_th:
            return True
        
        return False

    def get_traj_infos(self,):
        info_str = "loading traj_infos from {} ... ".format(self.traj_info_dir)
        if self.traj_len_th > 0:
            info_str += "filter out trajs with traj_len_th = {}".format(self.traj_len_th)
        if self.min_region_th > 0:
            info_str += " min_region_th = {}".format(self.min_region_th)
        print(info_str)

        traj_infos = dict()
        for seg_tag in tqdm(self.segment_tags):
            
            path = os.path.join(self.traj_info_dir,seg_tag + ".json")
            with open(path,'r') as f:
                tracking_results = json.load(f)

            res0 = tracking_results[0]

            has_cls =  "class" in res0.keys()
            has_tid = "tid" in res0.keys()   

            fstarts = []
            scores = []
            bboxes = []
            VinVL_clsids = []
            cls_ids = []
            tids = []
            ids_left = []
            for ii,res in enumerate(tracking_results):
                traj_len = len(res["bboxes"])
                h = max([b[3]-b[1] for b in res["bboxes"]])
                w = max([b[2]-b[0] for b in res["bboxes"]])
                if self.is_filter_out(h,w,traj_len):
                    continue
                
                fstarts.append(res["fstart"]) 
                scores.append(res["score"])
                bboxes.append(torch.as_tensor(res["bboxes"])) 

                ids_left.append(ii)
                if has_cls:
                    cls_ids.append(self.enti_cls2id[res["class"]]) 
                if has_tid:
                    tids.append(res["tid"])
                

            if ids_left: 
                ids_left = torch.as_tensor(ids_left)

                vit_feats = pickle.load(open(os.path.join(self.vit_feat_dir, seg_tag+'.pkl'), 'rb'))
                vit_feats = torch.from_numpy(vit_feats)
                if vit_feats.size(0) != ids_left.size(0):
                    vit_feats = vit_feats[ids_left]

                if self.filter_results is not None:
                    res = self.filter_results[seg_tag]
                    filter_ids = res['filter_ids']
                    bboxes = [bboxes[i] for i in filter_ids.tolist()]
                else:
                    lens = vit_feats.size(0)
                    filter_ids = torch.arange(start=0, end=lens, step=1, dtype=torch.long)
                traj_info = {
                    "fstarts":torch.as_tensor(fstarts)[filter_ids],
                    "scores":torch.as_tensor(scores)[filter_ids], 
                    "bboxes":bboxes,  
                    "ids_left":ids_left,
                    "vit_feats": vit_feats[filter_ids],
                }
                if has_cls:
                    traj_info.update({"cls_ids":torch.as_tensor(cls_ids)}) 

                if has_tid:
                    traj_info.update({"tids":torch.as_tensor(tids)})

            else:
                traj_info = None
            traj_infos[seg_tag] = traj_info

        return traj_infos

    def load_traj_infos(self,root_path=''):
        traj_infos = dict()

        for seg_tag in tqdm(self.segment_tags):

            path = os.path.join(root_path, seg_tag+".pkl")
            if not os.path.exists(path):
                traj_infos[seg_tag] = None
            else:
                traj_info = pickle.load(open(path, 'rb'))
                traj_infos[seg_tag] = traj_info
        return traj_infos

    def merge_gt_traj(self):
        gt_trajs = dict()
        traj_dir = self.gt_traj_track_dir
        vit_feat_dir = self.gt_vit_feat_path

        print("merge gt traj training data ...")
        for filename in tqdm(sorted(os.listdir(traj_dir))):
            seg_tag = filename.split('.')[0]
            
            path = os.path.join(traj_dir,filename)
            with open(path,'r') as f:
                gt_results = json.load(f)
            
            fstarts = []
            scores = []
            bboxes = []
            for res in gt_results:
                fstarts.append(res["fstart"]) 
                scores.append(res["score"])
                bboxes.append(torch.as_tensor(res["bboxes"]))
            
            if scores:  
                
                fstarts = torch.as_tensor(fstarts) 
                scores = torch.as_tensor(scores)  
                vit_feats = torch.from_numpy(pickle.load(open(os.path.join(vit_feat_dir, seg_tag+'.pkl'), 'rb')))

                gt_traj_info = {
                    "fstarts": fstarts,
                    "scores": scores,
                    "bboxes": bboxes,
                    "vit_feats": vit_feats
                }
                gt_trajs[seg_tag] = gt_traj_info

                if self.det_traj_infos[seg_tag] is not None:
                    det_traj_info = self.det_traj_infos[seg_tag] 
                    det_traj_info["fstarts"] = torch.cat([det_traj_info["fstarts"],fstarts],dim=0)
                    det_traj_info["scores"] = torch.cat([det_traj_info["scores"],scores],dim=0)
                    det_traj_info["vit_feats"] = torch.cat([det_traj_info["vit_feats"],vit_feats],dim=0)
                    det_traj_info["bboxes"] = det_traj_info["bboxes"] + bboxes
                    self.det_traj_infos[seg_tag] = det_traj_info
                else:
                    self.det_traj_infos[seg_tag] = gt_traj_info
            else:
                pass
        self.gt_traj_infos = gt_trajs

    def __len__(self):

        return len(self.segment_tags)


    def filter_segments(self):
        print("filter out segments with traj_cls_splits={}, pred_cls_splits={}".format(self.traj_cls_splits,self.pred_cls_splits))
        segment_tags_have_labels = []
        for seg_tag in self.segment_tags:
            video_name, seg_fs, seg_fe = seg_tag.split('-')  
            seg_fs,seg_fe = int(seg_fs),int(seg_fe)

            relations = self.video_annos[video_name]["relation_instances"]
            trajid2cls_map = {traj["tid"]:traj["category"] for traj in self.video_annos[video_name]["subject/objects"]}

            count = 0
            for rel in relations: 

                s_tid,o_tid = rel["subject_tid"],rel["object_tid"]
                s_cls,o_cls = trajid2cls_map[s_tid],trajid2cls_map[o_tid]
                s_split,o_split = self.enti_cls2split[s_cls],self.enti_cls2split[o_cls]
                p_cls = rel["predicate"]
                if not ((s_split in self.traj_cls_splits) and (o_split in self.traj_cls_splits)):
                    continue
                if not (self.pred_cls2split[p_cls] in self.pred_cls_splits):
                    continue

                fs,fe =  rel["begin_fid"],rel["end_fid"]  
                if not (seg_fs <= fs and fe <= seg_fe):  
                    continue

                count += 1
            if count == 0:
                continue
            segment_tags_have_labels.append(seg_tag)
        print("done. {} segments left".format(len(segment_tags_have_labels)))
        return segment_tags_have_labels

    def get_relative_position_feature(self, target='det_traj'):
        print("preparing relative position features ...")
        rel_pos_features = dict()
        for seg_tag in tqdm(self.segment_tags):

            if target =='det_traj':
                traj_fstarts = self.det_traj_infos[seg_tag]["fstarts"]  
                traj_bboxes = self.det_traj_infos[seg_tag]["bboxes"] 
            elif target == 'gt_traj':
                traj_fstarts = self.gt_traj_infos[seg_tag]["fstarts"]  
                traj_bboxes = self.gt_traj_infos[seg_tag]["bboxes"] 

            n_det = len(traj_bboxes)
            pair_ids = trajid2pairid(n_det)
            sids = pair_ids[:,0]
            oids = pair_ids[:,1]

            s_trajs = [traj_bboxes[idx] for idx in sids]  
            o_trajs = [traj_bboxes[idx] for idx in oids] 

            s_fstarts = traj_fstarts[sids]  
            o_fstarts = traj_fstarts[oids] 

            s_lens = torch.as_tensor([x.shape[0] for x in s_trajs],device=s_fstarts.device)
            o_lens = torch.as_tensor([x.shape[0] for x in o_trajs],device=o_fstarts.device)

            s_duras = torch.stack([s_fstarts,s_fstarts+s_lens],dim=-1) 
            o_duras = torch.stack([o_fstarts,o_fstarts+o_lens],dim=-1)  

            s_bboxes = [torch.stack((boxes[0,:],boxes[-1,:]),dim=0) for boxes in s_trajs]  
            s_bboxes = torch.stack(s_bboxes,dim=0) 

            o_bboxes = [torch.stack((boxes[0,:],boxes[-1,:]),dim=0) for boxes in o_trajs]
            o_bboxes = torch.stack(o_bboxes,dim=0)  

            subj_x, subj_y, subj_w, subj_h = _to_xywh(s_bboxes.float())  
            obj_x, obj_y, obj_w, obj_h = _to_xywh(o_bboxes.float())    

            log_subj_w, log_subj_h = torch.log(subj_w), torch.log(subj_h)
            log_obj_w, log_obj_h = torch.log(obj_w), torch.log(obj_h)

            rx = (subj_x-obj_x)/obj_w   
            ry = (subj_y-obj_y)/obj_h
            rw = log_subj_w-log_obj_w
            rh = log_subj_h-log_obj_h
            ra = log_subj_w+log_subj_h-log_obj_w-log_obj_h
            rt = (s_duras-o_duras) / 30  
            rel_pos_feat = torch.cat([rx,ry,rw,rh,ra,rt],dim=-1) 

            rel_pos_features[seg_tag] =  rel_pos_feat

        return rel_pos_features

    def __getitem__(self,idx):
        seg_tag = self.segment_tags[idx]
        det_traj_info = deepcopy(self.det_traj_infos[seg_tag]) 
        if self.gt_traj_infos is not None:
            gt_traj_infos = deepcopy(self.gt_traj_infos[seg_tag])
            gt_rel_pos_feats = deepcopy(self.gt_rel_pos_feats[seg_tag])
            gt_pred_labels = deepcopy(self.gt_labels[seg_tag])
        else:
            gt_traj_infos = None
            gt_rel_pos_feats = None
            gt_pred_labels = None
        try:
            union_path = os.path.join(self.traj_union_dir, seg_tag+'.pkl')
            union_feats = torch.tensor(pickle.load(open(union_path, 'rb')))
        except Exception as e:
            print(union_path)
            print(e)
        
        rel_pos_feat = deepcopy(self.rel_pos_features[seg_tag]) 
        if self.assign_label:
            labels = deepcopy(self.assigned_labels[seg_tag])  
            pred_labels = labels["predicate"] 
            so_labels = labels["entity"] 
        else:            
            pred_labels = None
            so_labels = None
        
        return seg_tag,det_traj_info,gt_traj_infos,union_feats,rel_pos_feat,gt_rel_pos_feats,pred_labels,gt_pred_labels

    
    def get_collator_func(self):
        
        def collator_func(batch_data):

            bsz = len(batch_data)
            num_rt_vals = len(batch_data[0])
            return_values = tuple([batch_data[bid][vid] for bid in range(bsz)] for vid in range(num_rt_vals))
            
            return return_values

        return collator_func
 

    def get_anno(self):
        raise NotImplementedError
    
    def load_video_annos(self):
        
        annos = dict()
        for split in self.dataset_splits:
            anno_dir = os.path.join(self.dataset_dir,split)
            for filename in sorted(os.listdir(anno_dir)):
                video_name = filename.split('.')[0] 
                path = os.path.join(anno_dir,filename)

                with open(path,'r') as f:
                    anno_per_video = json.load(f)  
                annos[video_name] = anno_per_video

        return annos


    def label_assignment(self):

        print("please use `tools/VidVRD_label_assignment.py` to pre-assign label and save as cache")
        raise NotImplementedError


    def _getitem_for_assign_label(self, idx):

        seg_tag = self.segment_tags[idx]
        video_name, seg_fs, seg_fe = seg_tag.split('-') 
        seg_fs,seg_fe = int(seg_fs),int(seg_fe)
        gt_traj_info = self.gt_traj_infos[seg_tag]
        

        gt_triplets = defaultdict(list)
        relations = self.video_annos[video_name]["relation_instances"]
        trajid2cls_map = {traj["tid"]:traj["category"] for traj in self.video_annos[video_name]["subject/objects"]}

        for rel in relations: 
            s_tid,o_tid = rel["subject_tid"],rel["object_tid"]
            s_cls,o_cls = trajid2cls_map[s_tid],trajid2cls_map[o_tid]
            s_split,o_split = self.enti_cls2split[s_cls],self.enti_cls2split[o_cls]
            p_cls = rel["predicate"]
            if not ((s_split in self.traj_cls_splits) and (o_split in self.traj_cls_splits)):
                continue
            if not (self.pred_cls2split[p_cls] in self.pred_cls_splits):
                continue           
            fs,fe =  rel["begin_fid"],rel["end_fid"]  
            if not (seg_fs <= fs and fe <= seg_fe): 
                continue
            assert seg_fs == fs and seg_fe == fe
            assert (s_tid in gt_traj_info.keys()) and (o_tid in gt_traj_info.keys()) 
            
            gt_triplets[(s_tid,o_tid)].append((s_cls,p_cls,o_cls))
        assert len(gt_triplets) > 0

        gt_s_trajs = []
        gt_o_trajs = []
        gt_s_fstarts = []
        gt_o_fstarts = []
        gt_pred_vecs = []
        gt_so_clsids = []
        for k,spo_cls_list in gt_triplets.items():
            s_tid,o_tid = k
            pred_list = [spo_cls[1] for spo_cls in spo_cls_list]
            s_cls,o_cls = spo_cls_list[0][0],spo_cls_list[0][2]
            gt_so_clsids.append(
                [self.enti_cls2id[s_cls],self.enti_cls2id[o_cls]]
            )

            s_traj = gt_traj_info[s_tid]
            o_traj = gt_traj_info[o_tid]
            s_fs = s_traj["fstarts"]  
            o_fs = o_traj["fstarts"]
            s_boxes = s_traj["bboxes"]
            o_boxes = o_traj["bboxes"]
            assert len(s_boxes) == 30 and len(o_boxes) == 30 and s_fs == seg_fs and o_fs == seg_fs
            multihot = torch.zeros(size=(self.pred_num_base+self.pred_num_novel+1,)) 
            for pred in pred_list:
                p_cls_id = self.pred_cls_split_info["cls2id"][pred]
                multihot[p_cls_id] = 1
            
            gt_s_trajs.append(s_boxes)
            gt_o_trajs.append(o_boxes)
            gt_s_fstarts.append(s_fs - seg_fs) 
            gt_o_fstarts.append(o_fs - seg_fs) 
            gt_pred_vecs.append(multihot)
        gt_s_fstarts = torch.as_tensor(gt_s_fstarts)  
        gt_o_fstarts = torch.as_tensor(gt_o_fstarts)
        gt_pred_vecs = torch.stack(gt_pred_vecs,dim=0) 
        gt_so_clsids = torch.as_tensor(gt_so_clsids) 
        n_gt_pair = gt_pred_vecs.shape[0]

        det_traj_info = self.det_traj_infos[seg_tag]
        det_trajs = det_traj_info["bboxes"]    
        det_fstarts = det_traj_info["fstarts"]  
        pair_ids = trajid2pairid(len(det_trajs))  
        s_ids = pair_ids[:,0]  
        o_ids = pair_ids[:,1]

        det_s_fstarts = det_fstarts[s_ids]
        det_o_fstarts = det_fstarts[o_ids]
        det_s_trajs = [det_trajs[idx] for idx in s_ids]
        det_o_trajs = [det_trajs[idx] for idx in o_ids]


        vpoi_s = vPoI_broadcast(det_s_trajs,gt_s_trajs,det_s_fstarts,gt_s_fstarts) 
        vpoi_o = vPoI_broadcast(det_o_trajs,gt_o_trajs,det_o_fstarts,gt_o_fstarts) 
        vpoi_mat = torch.minimum(vpoi_s,vpoi_o)

        max_vpois,gt_pair_ids = torch.max(vpoi_mat,dim=-1) 

        mask = max_vpois > self.vpoi_th 
        assigned_pred_labels = gt_pred_vecs[gt_pair_ids,:] 
        assigned_so_labels = gt_so_clsids[gt_pair_ids,:]  

        assigned_pred_labels[~mask,:] = 0  
        assigned_pred_labels[~mask,0] = 1 

        return seg_tag,assigned_pred_labels,assigned_so_labels,mask


class VidVRDUnifiedDataset_GIoU(VidVRDUnifiedDataset):
    def __init__(self, **kargs):
        super().__init__(**kargs)

        split_tag = "".join(self.dataset_splits)
        path_ = os.path.join(self.cache_dir, self.tag,"{}_VidVRD{}_rel_giou_th-{}-{}.pkl".format(self.cache_tag,split_tag,self.traj_len_th,self.min_region_th))
        if os.path.exists(path_):
            print("rel_giou loading from {}".format(path_))
            with open(path_,'rb') as f:
                rel_gious = pickle.load(f)
        else:
            print("no cache file found, ", end="")
            rel_gious = self.get_traj_pair_GIoU()
            with open(path_,'wb') as f:
                pickle.dump(rel_gious,f)
            print("rel_giou saved at {}".format(path_))
        self.rel_gious = rel_gious
        if "train" in self.dataset_splits:
            gt_path_ = os.path.join(self.cache_dir,self.tag,"{}_VidVRD{}_rel_giou_for_gt_th-{}-{}.pkl".format(self.cache_tag,split_tag,self.traj_len_th,self.min_region_th))
            if os.path.exists(gt_path_):
                print("rel_giou loading from {}".format(gt_path_))
                with open(gt_path_,'rb') as f:
                    gt_rel_gious = pickle.load(f)
            else:
                print("no cache file found, ", end="")
                gt_rel_gious = self.get_traj_pair_GIoU(False)
                with open(gt_path_,'wb') as f:
                    pickle.dump(gt_rel_gious,f)
                print("rel_giou saved at {}".format(gt_path_))
            self.gt_rel_gious = gt_rel_gious
        else:
            self.gt_rel_gious = None

    def get_traj_pair_GIoU(self, is_det=True):
        all_rel_gious = dict()
        print("preparing GIoUs ...")
        for seg_tag in tqdm(self.segment_tags):
            if is_det:
                traj_bboxes = self.det_traj_infos[seg_tag]["bboxes"] 
            else:
                traj_bboxes = self.gt_traj_infos[seg_tag]["bboxes"]
            n_det = len(traj_bboxes)
            pair_ids = trajid2pairid(n_det)
            sids = pair_ids[:,0]
            oids = pair_ids[:,1]
            n_pair = pair_ids.shape[0]

            s_trajs = [traj_bboxes[idx] for idx in sids]  
            o_trajs = [traj_bboxes[idx] for idx in oids] 

            start_s_box = torch.stack([boxes[0,:] for boxes in s_trajs],dim=0) 
            start_o_box = torch.stack([boxes[0,:] for boxes in o_trajs],dim=0) 

            end_s_box = torch.stack([boxes[-1,:] for boxes in s_trajs],dim=0)  
            end_o_box = torch.stack([boxes[-1,:] for boxes in o_trajs],dim=0) 

            start_giou = bbox_GIoU(start_s_box,start_o_box)[range(n_pair),range(n_pair)] 
            end_giou = bbox_GIoU(end_s_box,end_o_box)[range(n_pair),range(n_pair)] 
            se_giou = torch.stack([start_giou,end_giou],dim=-1)  

            all_rel_gious[seg_tag] = se_giou
        return all_rel_gious
    
    def __getitem__(self, idx):
       
        seg_tag,det_traj_info,gt_traj_infos,union_feats, rel_pos, gt_rel_pos,pred_labels,gt_pred_labels = super().__getitem__(idx)
        rel_giou = deepcopy(self.rel_gious[seg_tag])
        rel_pos_feat = (rel_pos,rel_giou)

        if self.gt_traj_infos is not None:
            gt_rel_giou = deepcopy(self.gt_rel_gious[seg_tag])
            gt_rel_pos_feat = (gt_rel_pos, gt_rel_giou)
        else:
            gt_rel_pos_feat = None

        return seg_tag,det_traj_info,gt_traj_infos,union_feats,rel_pos_feat,gt_rel_pos_feat,pred_labels,gt_pred_labels

