from enum import Enum

class FloodCategory(Enum):
    """Enumeration of flood categories based on max flood depth."""
    MODEL_TINY= '/home/users/li1995/global_flood/FloodRisk-DL/terratorch/output/all-tiny/checkpoints/best-epoch=18.ckpt'
    MODEL_100M = '/home/users/li1995/global_flood/FloodRisk-DL/terratorch/output/all-100/checkpoints/best-epoch=30.ckpt'
    MODEL_300M = '/home/users/li1995/global_flood/FloodRisk-DL/terratorch/output/all-300/checkpoints/best-epoch=29.ckpt'
    MODEL_600M = '/oak/stanford/groups/gorelick/ZhiLi/FloodRisk-DL/terratorch/output/all-600/checkpoints/best-epoch=15.ckpt'
    MODEL_RES50= '/home/users/li1995/global_flood/FloodRisk-DL/terratorch/output/all-unet-res50/checkpoints/best-epoch=51.ckpt'
    MODEL_RES101= '/home/users/li1995/global_flood/FloodRisk-DL/terratorch/output/all-unet-res101/checkpoints/best-epoch=49.ckpt'
    