import matplotlib.pyplot as plt
import compare_module.compare as cmp
VOC_BBOX_LABEL_NAMES = (
    'fly',
    'bike',
    'bird',
    'boat',
    'bott',
    'bus',
    'c',
    'cat',
    'chair',
    'cow',
    'table',
    'dog',
    'horse',
    'moto',
    'p',
    'plant',
    'shep',
    'sofa',
    'train',
    'tv',
    'bg',
)

COLORS = {
    cmp.TRUE_POS: 'green',
    cmp.ERR_LABEL: 'red',
    cmp.ERR_BBOX_SIZE: 'orange',
    cmp.ERR_BBOX_UNNES: 'purple',
    cmp.ERR_LACK_BBOX: 'yellow',
    cmp.ERR_REDUN: 'blue',
}


def draw_bbox(ax, bbox, label, color):
    if bbox is None:
        return ax
    bbox = bbox[0] if type(bbox[0]) == list else bbox
    xy = (bbox[1], bbox[0])
    height = bbox[2] - bbox[0]
    width = bbox[3] - bbox[1]
    ax.add_patch(plt.Rectangle(
            xy, width, height, fill=False, edgecolor=color, linewidth=2))
    if label is not None:
        ax.text(bbox[1], bbox[0],
                VOC_BBOX_LABEL_NAMES[label],
                style='italic',
                bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 0})
    return ax


def cust_vis_bbox(ax, cmp_res, figname):
    if len(cmp_res) == 0:
        return ax
    for res in cmp_res:
        if res['box_test'] is None:
            ax = draw_bbox(ax, res['box_mean'], res['label_mean'], COLORS[res['err']])
        ax = draw_bbox(ax, res['box_test'], res['label_test'], COLORS[res['err']])
    plt.savefig(figname)
    plt.close()
