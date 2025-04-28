
model_args_tiny = {
    "backbone_bands": ["BLUE", "NIR_BROAD", "SWIR_1"],
    "backbone": "terratorch_prithvi_eo_tiny",
    "backbone_pretrained": False,
    "backbone_img_size": 512,
    "backbone_patch_embed_cfg": {"in_channels": 3},  # Configure for 12-channel input
    "necks": [
        # {"name": "SelectIndices", "indices": [5, 11, 17, 23]},
        {"name": "ReshapeTokensToImage"},
        {"name": "LearnedInterpolateToPyramidal"}
    ],
    "decoder": "UNetDecoder",
    "decoder_channels": [512, 256, 128, 64],
    "head_dropout": 0.1,
    "num_classes": 5
}

model_args_100 = {
    "backbone_bands": ["BLUE", "NIR_BROAD", "SWIR_1"],
    "backbone": "terratorch_prithvi_eo_v1_100",
    "backbone_pretrained": True,
    "backbone_img_size": 512,
    "backbone_patch_embed_cfg": {"in_channels": 3},  # Configure for 12-channel input
    "necks": [
        {"name": "SelectIndices", "indices": [0, 1, 2, 3]},
        {"name": "ReshapeTokensToImage"},
        {"name": "LearnedInterpolateToPyramidal"}
    ],
    "decoder": "UNetDecoder",
    "decoder_channels": [512, 256, 128, 64],
    "head_dropout": 0.1,
    "num_classes": 5
}

model_args_300 = {
    "backbone_bands": ["BLUE", "NIR_BROAD", "SWIR_1"],
    "backbone": "terratorch_prithvi_eo_v2_300",
    "backbone_pretrained": True,
    "backbone_img_size": 512,
    "backbone_patch_embed_cfg": {"in_channels": 3},  # Configure for 12-channel input
    "necks": [
        {"name": "SelectIndices", "indices": [5, 11, 17, 23]},
        {"name": "ReshapeTokensToImage"},
        {"name": "LearnedInterpolateToPyramidal"}
    ],
    "decoder": "UNetDecoder",
    "decoder_channels": [512, 256, 128, 64],
    "head_dropout": 0.1,
    "num_classes": 5
}

model_args_600 = {
    "backbone_bands": ["BLUE", "NIR_BROAD", "SWIR_1"],
    "backbone": "terratorch_prithvi_eo_v2_600",
    "backbone_pretrained": True,
    "backbone_img_size": 512,
    "backbone_patch_embed_cfg": {"in_channels": 3},  # Configure for 12-channel input
    "necks": [
        {"name": "SelectIndices", "indices": [5, 11, 17, 23]},
        {"name": "ReshapeTokensToImage"},
        {"name": "LearnedInterpolateToPyramidal"}
    ],
    "decoder": "UNetDecoder",
    "decoder_channels": [512, 256, 128, 64],
    "head_dropout": 0.1,
    "num_classes": 5
}

model_args_unet = {
        "backbone":"resnet152", # see smp_encoders.keys()
        'model': 'UnetPlusPlus', # 'DeepLabV3', 'DeepLabV3Plus', 'FPN', 'Linknet', 'MAnet', 'PAN', 'PSPNet', 'Unet', 'UnetPlusPlus' 
        "bands": ["BLUE", "NIR_BROAD", "SWIR_1"],
        "in_channels": 3,
        "num_classes": 5,
        "pretrained": False,
}

