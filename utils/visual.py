# coding:UTF-8
import os
import numpy as np
import cv2


def gen_hsv_class():
    return np.array([
        0,  # road          aer
        0.05,  # sidewalk      bike
        0.1,  # building      bird
        0.15,  # wall          boat
        0.2,  # fence         bottle
        0.25,  # pole bus      bus
        0.3,  # traffic_light car
        0.35,  # traffic_sign  cat
        0.4,  # vegetation    chair
        0.45,  # terrain       cow
        # ----------------------------
        0.5,  # sky           table
        0.55,  # person        dog
        0.6,  # person        horse
        0.65,  # car           motorbike
        0.7,  # truck         person
        0.75,  # bus           potting
        0.8,  # train         sheep
        0.85,  # motorcycle    sofa
        0.9,  # bike          train
        1  # xxxx          tv
    ]) * 255


def process_origin_image(image):
    rgb_image = np.transpose(image, axes=[1, 2, 0])
    rgb_image += np.array([104.008, 116.669, 122.675])
    rgb_image = np.array(rgb_image, dtype=np.uint8)
    return rgb_image


def process_no_semantic_edge(single_label):
    single_label = np.amax(single_label, axis=0)
    single_label[single_label == 255] = 0
    no_semantic_edge = np.zeros((single_label.shape[0], single_label.shape[1], 3), dtype=np.uint8)
    no_semantic_edge[:, :, 0] = (single_label * 255).astype(np.uint8)
    no_semantic_edge[:, :, 1] = (single_label * 255).astype(np.uint8)
    no_semantic_edge[:, :, 2] = (single_label * 255).astype(np.uint8)
    return no_semantic_edge


def process_semantic_edge(prob, thresh=0.6):
    prob[prob == 255] = 0
    prob = np.transpose(prob, axes=[1, 2, 0])
    hsv_class = gen_hsv_class()
    # 阈值筛选
    prob[prob <= thresh] = 0
    rows = prob.shape[0]
    cols = prob.shape[1]
    chns = prob.shape[2]
    ii, jj = np.meshgrid(range(0, rows), range(0, cols), indexing='ij')
    prob_out = np.zeros(prob.shape, dtype=np.float32)
    # 选择概率最高的两个
    for ik in range(0, 2):
        idx_max = np.argmax(prob, axis=2)
        prob_out[ii, jj, idx_max] = prob[ii, jj, idx_max]
        prob[ii, jj, idx_max] = -1
    # 核心代码：多标签可视化
    label_hsv = np.zeros((rows, cols, 3), dtype=np.float32)
    prob_sum = np.zeros((rows, cols), dtype=np.float32)  # 决定颜色，颜色和求平均
    prob_edge = np.zeros((rows, rows), dtype=np.float32)  # 决定饱和度，由概率确定
    for k in range(0, chns):
        prob_k = prob_out[:, :, k].astype(np.float32)
        if prob_k.max() == 0:
            continue
        hi = hsv_class[k]
        # 1.色度（求平均）
        label_hsv[:, :, 0] += prob_k * hi  # H
        prob_sum += prob_k
        prob_edge = np.maximum(prob_edge, prob_k)

    prob_sum[prob_sum == 0] = 1.0
    label_hsv[:, :, 0] /= prob_sum
    # 2.饱和度
    label_hsv[:, :, 1] = prob_edge * 255
    # 3.明度
    label_hsv[:, :, 2] = 255
    # 由HSV转换成BGR
    label_bgr = cv2.cvtColor(label_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return label_bgr


# ========================================================CASENet数据展示 ======================================================
def visual_data1(image, label, prob1, prob2, image_name='test', is_show=True, dir='./result/'):
    image = process_origin_image(image)
    # no_semantic_edge = process_no_semantic_edge(label)
    semantic_edge = process_semantic_edge(label)
    semantic_prob1 = process_semantic_edge(prob1, thresh=0.6)
    semantic_prob2 = process_semantic_edge(prob2, thresh=0.6)

    htitch1 = np.hstack((image, semantic_edge))
    htitch2 = np.hstack((semantic_prob1, semantic_prob2))
    vtitch = np.vstack((htitch1, htitch2))

    if is_show:
        cv2.imshow("train_data", vtitch)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        if not os.path.exists(dir):
            os.makedirs(dir)
        cv2.imwrite(dir + image_name, vtitch)


# ========================================================SC-SED数据展示=======================================================
def visual_data2(image, label, prob, image_name='test', is_show=True, dir='./result/'):
    image = process_origin_image(image)
    # no_semantic_edge = process_no_semantic_edge(label)
    semantic_edge = process_semantic_edge(label)
    semantic_prob = process_semantic_edge(prob, thresh=0.2)

    htitch = np.hstack((image, semantic_edge, semantic_prob))

    if is_show:
        cv2.imshow("train_data", htitch)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        if not os.path.exists(dir):
            os.makedirs(dir)
        cv2.imwrite(dir + image_name, htitch)

# ========================================================BQH数据展示=======================================================
def apply_mask(image, mask, color):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] + color[c],
                                  image[:, :, c])
    return image


def visualize_prediction(dataset, pred):
    n, h, w = pred.shape
    image = np.zeros((h, w, 3))
    # image = image.astype(np.uint32)

    if dataset == 'cityscapes':
        colors = [[128, 64, 128],
                  [244, 35, 232],
                  [70, 70, 70],
                  [102, 102, 156],
                  [190, 153, 153],
                  [153, 153, 153],
                  [250, 170, 30],
                  [220, 220, 0],
                  [107, 142, 35],
                  [152, 251, 152],
                  [70, 130, 180],
                  [220, 20, 60],
                  [255, 0, 0],
                  [0, 0, 142],
                  [0, 0, 70],
                  [0, 60, 100],
                  [0, 80, 100],
                  [0, 0, 230],
                  [119, 11, 32]]
    elif dataset == 'ade20k':
        from utils.ade20k.ade20k_stuff import stuff_colors
        colors = stuff_colors
    else:
        assert dataset == 'sbd'
        colors = [[128, 0, 0],
                  [0, 128, 0],
                  [128, 128, 0],
                  [0, 0, 128],
                  [128, 0, 128],
                  [0, 128, 128],
                  [128, 128, 128],
                  [64, 0, 0],
                  [192, 0, 0],
                  [64, 128, 0],
                  [192, 128, 0],
                  [64, 0, 128],
                  [192, 0, 128],
                  [64, 128, 128],
                  [192, 128, 128],
                  [0, 64, 0],
                  [128, 64, 0],
                  [0, 192, 0],
                  [128, 192, 0],
                  [0, 64, 128],
                  [255, 255, 255]]

    pred[pred == 255] = 0
    pred = np.where(pred >= 0.2, 1, 0).astype(bool)
    edge_sum = np.zeros((h, w))

    for i in range(n):
        color = colors[i]
        edge = pred[i, :, :]
        edge_sum = edge_sum + edge
        masked_image = apply_mask(image, edge, color)

    edge_sum = np.array([edge_sum, edge_sum, edge_sum])
    edge_sum = np.transpose(edge_sum, (1, 2, 0))
    idx = edge_sum > 0
    masked_image[idx] = masked_image[idx] / edge_sum[idx]
    masked_image[~idx] = 255

    return masked_image


def label_img_to_color(dataset, img):
    if dataset == 'cityscapes':
        label_to_color = {
            0: [128, 64, 128],
            1: [244, 35, 232],
            2: [70, 70, 70],
            3: [102, 102, 156],
            4: [190, 153, 153],
            5: [153, 153, 153],
            6: [250, 170, 30],
            7: [220, 220, 0],
            8: [107, 142, 35],
            9: [152, 251, 152],
            10: [70, 130, 180],
            11: [220, 20, 60],
            12: [255, 0, 0],
            13: [0, 0, 142],
            14: [0, 0, 70],
            15: [0, 60, 100],
            16: [0, 80, 100],
            17: [0, 0, 230],
            18: [119, 11, 32]}
    elif dataset == "ade20k":
        from utils.ade20k.ade20k_stuff import label2colors
        label_to_color = label2colors
    else:
        assert dataset == 'sbd'
        label_to_color = {
            0: [128, 0, 0],
            1: [0, 128, 0],
            2: [128, 128, 0],
            3: [0, 0, 128],
            4: [128, 0, 128],
            5: [0, 128, 128],
            6: [128, 128, 128],
            7: [64, 0, 0],
            8: [192, 0, 0],
            9: [64, 128, 0],
            10: [192, 128, 0],
            11: [64, 0, 128],
            12: [192, 0, 128],
            13: [64, 128, 128],
            14: [192, 128, 128],
            15: [0, 64, 0],
            16: [128, 64, 0],
            17: [0, 192, 0],
            18: [128, 192, 0],
            19: [0, 64, 128],
            20: [255, 255, 255]}

    img_height, img_width = img.shape

    img_color = np.zeros((img_height, img_width, 3))
    for row in range(img_height):
        for col in range(img_width):
            label = img[row, col]
            img_color[row, col] = np.array(label_to_color[label])
    return img_color

def visual_data_seg(dataset, image, label, prob, prob_seg, image_name='test', is_show=True, dir='./result/'):
    image = process_origin_image(image)
    gt = visualize_prediction(dataset, label)
    prob = visualize_prediction(dataset, prob)
    prob_seg = label_img_to_color(dataset, prob_seg)
    # prob_seg = cv2.resize(prob_seg, (prob_seg.shape[0]*2, prob_seg.shape[1]*2), interpolation=cv2.INTER_NEAREST)

    htitch1 = np.hstack((image, gt))
    htitch2 = np.hstack((prob_seg, prob))
    htitch = np.vstack((htitch1, htitch2))

    if is_show:
        cv2.imshow("train_data", htitch)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        if not os.path.exists(dir):
            os.makedirs(dir)
        cv2.imwrite(dir + image_name, htitch)


def visual_data3(dataset, image, label, prob, prob_seg, prob_edge, image_name='test', is_show=True, dir='./result/'):
    image = process_origin_image(image)
    gt = visualize_prediction(dataset, label)
    prob = visualize_prediction(dataset, prob)
    # prob_seg = visualize_prediction(dataset, prob_seg)
    prob_seg = label_img_to_color(dataset, prob_seg)
    # prob_seg = cv2.resize(prob_seg, (prob_seg.shape[0]*2, prob_seg.shape[1]*2), interpolation=cv2.INTER_NEAREST)
    prob_edge = 255 - 255*prob_edge
    # prob_edge = cv2.resize(prob_edge, (prob_edge.shape[0] * 2, prob_edge.shape[1] * 2), interpolation=cv2.INTER_NEAREST)
    prob_edge = np.expand_dims(prob_edge, 2).repeat(3, axis=2)

    htitch1 = np.hstack((image, gt))
    htitch2 = np.hstack((prob_seg, prob))
    htitch3 = np.hstack((prob_edge, prob_edge))
    htitch = np.vstack((htitch1, htitch2, htitch3))

    if is_show:
        cv2.imshow("train_data", htitch)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        if not os.path.exists(dir):
            os.makedirs(dir)
        cv2.imwrite(dir + "/" +image_name, htitch)

def visual_data4(dataset, image, label, prob, prob_seg0, prob_seg1, prob_edge, image_name='test', is_show=True, dir='./result/'):
    image = process_origin_image(image)
    gt = visualize_prediction(dataset, label)
    prob = visualize_prediction(dataset, prob)
    prob_seg0 = label_img_to_color(dataset, prob_seg0)
    prob_seg1 = label_img_to_color(dataset, prob_seg1)
    # prob_seg = cv2.resize(prob_seg, (prob_seg.shape[0]*2, prob_seg.shape[1]*2), interpolation=cv2.INTER_NEAREST)
    prob_edge = 255 - 255*prob_edge
    # prob_edge = cv2.resize(prob_edge, (prob_edge.shape[0] * 2, prob_edge.shape[1] * 2), interpolation=cv2.INTER_NEAREST)
    prob_edge = np.expand_dims(prob_edge, 2).repeat(3, axis=2)

    htitch1 = np.hstack((image, gt))
    htitch2 = np.hstack((prob_seg0, prob))
    htitch3 = np.hstack((prob_seg1, prob_edge))
    htitch = np.vstack((htitch1, htitch2, htitch3))

    if is_show:
        cv2.imshow("train_data", htitch)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        if not os.path.exists(dir):
            os.makedirs(dir)
        cv2.imwrite(dir + image_name, htitch)

def visual_only_sed(dataset, image, label, prob, prob_aux=None, image_name='test', is_show=True, dir='./result/'):
    image = process_origin_image(image)
    gt = visualize_prediction(dataset, label)
    prob = visualize_prediction(dataset, prob)
    prob_aux = None if prob_aux is None else visualize_prediction(dataset, prob_aux)
    if prob_aux is None:
        htitch = np.hstack((image, gt, prob))
    else:
        htitch1 = np.hstack((image, gt))
        htitch2 = np.hstack((prob_aux, prob))
        htitch = np.vstack((htitch1, htitch2))
    if is_show:
        cv2.imshow("train_data", htitch)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        if not os.path.exists(dir):
            os.makedirs(dir)
        cv2.imwrite(dir + image_name, htitch)

def visual_data5(dataset, image, label, prob, prob_seg, prob_edge, image_name='test', is_show=True, dir='./result/'):
    image = process_origin_image(image)
    gt = visualize_prediction(dataset, label)
    prob = visualize_prediction(dataset, prob)
    prob_seg = label_img_to_color(dataset, prob_seg)
    # prob_seg = cv2.resize(prob_seg, (prob_seg.shape[0]*2, prob_seg.shape[1]*2), interpolation=cv2.INTER_NEAREST)
    prob_edge = 255 - 255*prob_edge
    # prob_edge = cv2.resize(prob_edge, (prob_edge.shape[0] * 2, prob_edge.shape[1] * 2), interpolation=cv2.INTER_NEAREST)
    prob_edge = np.expand_dims(prob_edge, 2).repeat(3, axis=2)

    htitch = np.hstack((image, gt, prob, prob_edge, prob_seg))

    if is_show:
        cv2.imshow("train_data", htitch)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        if not os.path.exists(dir):
            os.makedirs(dir)
        cv2.imwrite(dir + image_name, htitch)