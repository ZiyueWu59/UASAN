model_pred_cfg = dict(
    num_base = 71,
    num_novel = 61,
    temperature =  0.02125491015613079, 
    n_context_tokens = 10,
    prompter_ckpt_path = '', 
    pred_cls_split_info_path = "configs/VidVRD_pred_class_spilt_info_v2.json",
    finetune_path = "pre_train.pth",
    teacher_path = "teacher_model.pth",
)

model_traj_cfg = dict(
    num_base = 25,
    num_novel = 10,
    text_emb_path_bert = "OpenVidSGG/vidvrd_bert_prompt_proj_ObjTextEmbeddings.pth", 
    temperature_init = 0.02125491015613079, 
    traj_former_cfg = dict(
        num_query_token=32, 
    ),
)


eval_dataset_cfg = dict(
    dataset_splits = ("test",),
    enti_cls_spilt_info_path = "configs/VidVRD_class_spilt_info.json",
    pred_cls_split_info_path = "configs/VidVRD_pred_class_spilt_info_v2.json",
    dataset_dir = "OpenVidSGG/VidVRD_VidOR/vidvrd-dataset",
    traj_info_dir = "OpenVidSGG/VidVRD-II/tracklets_results/VidVRD_segment30_tracking_results",
    vit_feat_dir = "OpenVidSGG/Extract/scene_graph_benchmark/output/vit/VidVRD_traj_features_seg30",
    traj_union_dir = "OpenVidSGG/Extract/scene_graph_benchmark/output/vit/VidVRD_union_frame_features_seg30",
    cache_dir = "OpenVidSGG/cache_vidvrd_v1",
    gt_training_traj_supp = None,
    traj_len_th = 15,
    min_region_th = 5,
)

GTeval_dataset_cfg = dict(
    dataset_splits = ("test",),
    enti_cls_spilt_info_path = "configs/VidVRD_class_spilt_info.json",
    pred_cls_split_info_path = "configs/VidVRD_pred_class_spilt_info_v2.json",
    dataset_dir = "OpenVidSGG/VidVRD_VidOR/vidvrd-dataset",
    traj_info_dir = "OpenVidSGG/scene_graph_benchmark/output/VidVRDtest_tracking_results_gt",
    vit_feat_dir = 'OpenVidSGG/Extract/scene_graph_benchmark/output/vit/VidVRDtest_gt_traj_features_seg30',
    traj_union_dir = "OpenVidSGG/Extract/scene_graph_benchmark/output/vit/VidVRD_union_frame_features_seg30_gt",
    cache_dir = "OpenVidSGG/cache_vidvrd_v1",
    gt_training_traj_supp = None,
    traj_len_th = -1,
    min_region_th = -1,
    cache_tag = "gtbbox"
)


eval_cfg = dict(
    pred_topk = 10,
    return_triplets_topk = 200,
)

association_cfg = dict(
    inference_topk = eval_cfg["return_triplets_topk"], 
    association_algorithm = "greedy",
    association_linkage_threshold = 0.8,
    association_nms = 0.8,
    association_topk = 200,
    association_n_workers = 12
)
