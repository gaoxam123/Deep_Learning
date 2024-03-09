from torch_mtcnn import detect_faces
from PIL import Image
from PIL import ImageDraw

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle


def draw_facebox(image, bboxes):
    plt.imshow(image)
    ax = plt.gca()

    for box in bboxes:
        x1, y1, x2, y2 = [box[i] for i in range(4)]
        scores = box[4]
        w, h = x2 - x1 + 1, y2 - y1 + 1
        rect = plt.Rectangle((x1, y1), w, h, fill=False, color='green')
        ax.add_patch(rect)

        plt.text(x1, y1, s=f"{round((scores * 100), 2)}%", color="white", fontsize=7)
        
    plt.axis('off')
    plt.show()

image = Image.open('image.jpg')
bounding_boxes, landmarks = detect_faces(image)
draw_facebox(image, bounding_boxes)