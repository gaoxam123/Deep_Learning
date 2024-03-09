import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import config
import numpy as np

def cells_to_boxes(predictions, anchors, S, is_pred=True):
    batch_size = predictions.shape[0]
    # num_anchors = len(anchors)
    box_predictions = predictions[..., 1:5]

    if is_pred:
        confidence_scores = torch.sigmoid(predictions[..., 0:1])
        class_label = torch.argmax(predictions[..., 5:], dim=-1).unsqueeze(-1)
        anchors = anchors.reshape(1, len(anchors), 1, 1, 2)
        box_predictions[..., 0:2] = 2 * torch.sigmoid(box_predictions[..., 0:2]) - 0.5
        box_predictions[..., 2:4] = anchors * torch.square(2 * torch.sigmoid(box_predictions[..., 2:4]))

    else:
        confidence_scores = predictions[..., 0:1]
        class_label = predictions[..., 5:6]

    cell_indices = torch.arange(S).repeat(predictions.shape[0], 3, S, 1).unsqueeze(-1).to(predictions.device)
    x = 1 / S * (box_predictions[..., 0:1] + cell_indices)
    y = 1 / S * (box_predictions[..., 1:2] + cell_indices.permute(0, 1, 3, 2, 4))
    wh = 1 / S * box_predictions[..., 2:4]
    converted_boxes = torch.cat([class_label, confidence_scores, x, y, wh], dim=-1).reshape(batch_size, -1, 6)

    return converted_boxes.tolist()

def plot_image(image, boxes):
    cmap = plt.get_cmap('tab20b')
    class_labels = config.COCO_LABELS
    colors = [cmap(i) for i in np.linspace(0, 1, len(class_labels))]
    im = np.array(image)
    h, w, _ = im.shape

    fig, ax = plt.subplots(1)
    ax.imshow(im)

    for box in boxes:
        assert len(box) == 6
        class_pred = box[0]
        box = box[2:]
        conf_score = box[1]
        x1 = box[0] - box[2] / 2
        y1 = box[1] - box[3] / 2
        rect = patches.Rectangle((x1 * w, y1 * h),
                                 box[2] * w,
                                 box[3] * h,
                                linewidth=2,
                                edgecolor=colors[int(class_pred)],
                                facecolor='none')
        ax.add_patch(rect)
        plt.text(x1 * w, y1 * h, s=class_labels[int(class_pred)], color='white', verticalalignment='top', bbox={"color": colors[int(class_pred)], "pad": 0})

    plt.show()        