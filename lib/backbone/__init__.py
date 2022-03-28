from lib.backbone.vgg import Vgg13

BACKBONES = {
    "vgg13": Vgg13,
}


def get_backbone(name: str):
    """Get backbone given the name"""
    if name not in BACKBONES.keys():
        raise ValueError(
            f"Backbone {name} not in backbone list. Valid backbonesa are {BACKBONES.keys()}"
        )
    print(f"=> backbone: {name}")
    return BACKBONES[name]
