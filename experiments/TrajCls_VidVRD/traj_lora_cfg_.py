model_cfg = dict(
    num_base = 25,
    num_novel = 10,
    text_emb_path_bert = "OpenVidSGG/vidvrd_bert_prompt_proj_ObjTextEmbeddings.pth", 
    temperature_init = 0.02125491015613079, 
    traj_former_cfg = dict(
        num_query_token=32, 
    ),
)

eval_dataset_cfg = dict(
    class_splits = ("novel",),
    dataset_split = "test",
    class_spilt_info_path = "configs/VidVRD_class_spilt_info.json",
    dataset_dir = "OpenVidSGG/VidVRD_VidOR/vidvrd-dataset",
    tracking_res_dir = "OpenVidSGG/VidVRD-II/tracklets_results/VidVRD_segment30_tracking_results_th-15-5",
    vit_path = "OpenVidSGG/Extract/scene_graph_benchmark/output/vit/VidVRD_traj_features_seg30_th-15-5",
    vIoU_th = 0.5,
    cache_dir = "OpenVidSGG/cache_vidvrd"
)

GTeval_dataset_cfg = dict(
    class_splits = ("novel",),
    dataset_split = "test",
    class_spilt_info_path = "configs/VidVRD_class_spilt_info.json",
    dataset_dir = "OpenVidSGG/VidVRD_VidOR/vidvrd-dataset",
    tracking_res_dir = "OpenVidSGG/scene_graph_benchmark/output/VidVRDtest_tracking_results_gt",
    vit_path = "OpenVidSGG/Extract/scene_graph_benchmark/output/vit/VidVRDtest_gt_traj_features_seg30",
    vIoU_th = 0.5,
    cache_dir = "OpenVidSGG/cache_vidvrd"
)

eval_cfg = dict(
    vIoU_th = 0.5,
    batch_size = 16,
)