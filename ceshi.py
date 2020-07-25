# from PIL import Image
import numpy as np
import os
import json
import io
import requests
import cv2

# The thresholds used for detection
Orb_threshold = 0.4  # 0~1 float
Hamming_threshold = 60  # 0~1024 int

# The thresholds used for judgment
Distance_threshold = 50  # 0~100 int
# PreImageUrl,  CurImageUrl, PreImageClips, CurImageClips
# string, string, float, float



def get_image_clips(url):
    req = requests.get(url)
    imgs_data = json.loads(req.content)

    imgs = []
    for data in imgs_data['data']:
        source_img_url = data['可见光图片']
        source_img_bytes = requests.get(source_img_url).content

        source_img = cv2.imdecode(np.array(bytearray(source_img_bytes), np.uint8), -1)
        source_img = cv2.cvtColor(source_img, cv2.COLOR_BGR2GRAY)
        # cv2.imwrite(os.path.join(os.getcwd(), 'ceshi1.jpg'), source_img)
        # source_img = Image.open(io.BytesIO(source_img_bytes))
        # source_img.save(os.path.join(os.getcwd(),'ceshi1.jpg'),'jpeg')

        img_clips = []
        for coord in data['坐标']:
            [x1, y1, x2, y2] = map(int, [coord['左上X'], coord['左上Y'], coord['右下X'], coord['右下Y']])
            clip = source_img[y1:y2, x1:x2]
            img_clips.append({'clip': clip, 'coord': [x1, y1, x2, y2]})

        imgs.append({'source_img': source_img, 'img_clips': img_clips})

    return imgs


def get_expand_clip_templates(shape, img_template):
    [height, width] = shape
    [height_expand, width_expand] = map(int, [height / 3, width / 3])

    img_raw = img_template['source_img']
    [raw_height, raw_width] = img_raw.shape

    templates = []
    for img_clip in img_template['img_clips']:
        clip = img_clip['clip']
        [clip_height, clip_width] = img_clip['clip'].shape
        img_resize = cv2.resize(img_raw, (int(raw_width * width / clip_width), int(raw_height * height / clip_height)),
                                interpolation=cv2.INTER_CUBIC)
        [x1, y1, x2, y2] = img_clip['coord']
        [x1, y1, x2, y2] = [int(x1 * width / clip_width), int(y1 * height / clip_height), int(x2 * width / clip_width),
                            int(y2 * height / clip_height)]
        [x1, y1, x2, y2] = [max(0, x1 - width_expand), max(0, y1 - height_expand),
                            min(x2 + width_expand, raw_width * width / clip_width),
                            min(y2 + height_expand, raw_height * height / clip_height)]
        clip_template = img_resize[y1:y2, x1:x2]
        templates.append(clip_template)
    return templates


def OrbSimilarity(img_1, img_2):  # pillow格式
    # res = cv2.matchTemplate(source,template,method=cv2.TM_CCOEFF)
    # min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    # img_1, img_2 = clip_source, clip_template
    orb = cv2.ORB_create()
    keypoint1, descriptor1 = orb.detectAndCompute(img_1, None)
    keypoint2, descriptor2 = orb.detectAndCompute(img_2, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.knnMatch(descriptor1, descriptor2, k=1)

    max_dist = 0
    min_dist = 100
    good_num = 0
    for i in matches:
        min_dist = min(min_dist, i[0].distance) if i != [] else min_dist
        max_dist = max(max_dist, i[0].distance) if i != [] else max_dist
        good_num = good_num + 1 if i != [] and i[0].distance <= Distance_threshold else good_num
    if np.count_nonzero(matches) == 0:
        return 0
    similary = float(good_num) / np.count_nonzero(matches)
    return similary


def HammingDistace(img_1, img_2):  # opencv types
    featureImage = cv2.resize(img_1, (32, 32))
    hashcode1 = (featureImage >= np.mean(np.mean(featureImage))).astype(np.int8)
    featureImage = cv2.resize(img_2, (32, 32))
    hashcode2 = (featureImage >= np.mean(np.mean(featureImage))).astype(np.int8)
    return np.count_nonzero(hashcode1 != hashcode2)


def compare_To(img1, img2):
    img1_raw = img1['source_img']
    similarity = []
    hamming_distance = []
    [raw_height, raw_width] = img1_raw.shape
    for clip1 in img1['img_clips']:
        clip_shape = clip1['clip'].shape
        [height, width] = clip_shape
        [height_expand, width_expand] = map(int, [height / 3, width / 3])
        [x1, y1, x2, y2] = clip1['coord']
        [x1, y1, x2, y2] = [max(0, x1 - width_expand), max(0, y1 - height_expand), min(x2 + width_expand, raw_width),
                            min(y2 + height_expand, raw_height)]
        clip_source = img1_raw[y1:y2, x1:x2]
        clip_templates = get_expand_clip_templates(clip_shape, img2)

        max_similarity = 0
        min_hamming_dis = 1024
        for clip_template in clip_templates:
            max_similarity = max(max_similarity, OrbSimilarity(clip_source, clip_template))
            min_hamming_dis = min(min_hamming_dis, HammingDistace(clip_source, clip_template))
        # cv2.imwrite(os.path.join(os.getcwd(), 'ceshi1.jpg'), clip_template)
        similarity.append(max_similarity)
        hamming_distance.append(min_hamming_dis)
    return min(similarity) > Orb_threshold or max(hamming_distance) < Hamming_threshold


def is_Duplicated(img1, img2):
    if len(img1['img_clips']) != len(img2['img_clips']):
        return False
    img1_to_img2 = compare_To(img1, img2)
    img2_to_img1 = compare_To(img2, img1)
    return img1_to_img2 and img2_to_img1


if __name__ == '__main__':
    url = 'https://infrared.hzncc.cn/api/infrared/alarm/list/byNumber/3'
    imgs = get_image_clips(url)
    img1, img2 = imgs[0], imgs[1]
    cv2.imwrite(os.path.join(os.getcwd(), 'img0.jpg'), img1['source_img'])
    cv2.imwrite(os.path.join(os.getcwd(), 'img1.jpg'), img2['source_img'])
    cv2.imwrite(os.path.join(os.getcwd(), 'img0clip.jpg'), img1['img_clips'][0]['clip'])
    cv2.imwrite(os.path.join(os.getcwd(), 'img1clip.jpg'), img2['img_clips'][0]['clip'])
    print(is_Duplicated(img1, img2))
