"""
Utils for clip model
"""
from cn_clip.clip import load_from_name


finetune_dataset = "flickr30kcn_finetune_vit-l-14_roberta-base_batchsize64_1gpu"
PT_FILE = f"../fmhData/experiments/{finetune_dataset}/checkpoints/epoch_latest.pt"
def load_model(filename=PT_FILE, for_inference=True):
    """Load the finetuned cn-clip model"""
    model, preprocess = load_from_name(
        filename,
        # NOTE 实际用的模型是"ViT-L-14-336"，但在finetune时指定的名称为"ViT-L-14"
        # 导致在使用 "ViT-L-14-336" 进行本地模型加载时报错（？？）对。已重新微调。
        #  vision_model_name="ViT-L-14",
        vision_model_name="ViT-L-14-336",
        text_model_name="RoBERTa-wwm-ext-base-chinese",
        input_resolution=336,
    )
    if for_inference:
        model.eval()
        print("Load the model for inference (`model.eval()`)")
    return model, preprocess
