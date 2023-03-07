from .nms import mean_bbox
IOU_TRESHOLD = 0.35

ERR_BBOX_UNNES = -5
ERR_LACK_BBOX = -4
ERR_REDUN = -3
ERR_BBOX_SIZE = -2
ERR_LABEL = -1
TRUE_POS = 0

FUNC = mean_bbox
