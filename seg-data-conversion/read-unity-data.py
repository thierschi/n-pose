class UnitySegmentationInstance:
    def __init__(self, label_id: int, label_name: str, color: (int, int, int, int)):
        self.label_id = label_id
        self.label_name = label_name
        self.color = color


class UnityCapture:
    def __init__(self, image: str, segmentation_mask: str, instances: list[UnitySegmentationInstance]):
        self.image = image
        self.segmentation_mask = segmentation_mask
        self.instances = instances
