import matplotlib.pyplot as plt
import compare_module.utils as c
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
    c.TRUE_POS: 'green',
    c.ERR_LABEL: 'red',
    c.ERR_BBOX_SIZE: 'orange',
    c.ERR_BBOX_UNNES: 'purple',
    c.ERR_LACK_BBOX: 'yellow',
}


def draw_bbox(ax, bbox, label, color, is_mean):
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
                f'{"MEAN" if is_mean else "TEST"}: {VOC_BBOX_LABEL_NAMES[int(label)]}',
                style='italic',
                bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 0})
    return ax


def cust_vis_bbox(ax, cmp_res, figname):
    if len(cmp_res) == 0:
        return ax
    for res in cmp_res:
        ax = draw_bbox(ax, res['box_mean'], res['label_mean'], 'yellow', True)
        ax = draw_bbox(ax, res['box_test'], res['label_test'], 'green', False)
    plt.savefig(figname)
    plt.close()
