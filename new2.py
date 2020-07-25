# from PIL import Image
import numpy as np
import os
import json
import io
import requests
import cv2

# The thresholds used for detection
Orb_threshold = 0.4  # 0~1 float
Hamming_threshold = 60  # 0~64 int
Cosine_threshold = 200  # 0~1024 int

# The thresholds used for judgment
Distance_threshold = 50  # 0~100 int


# PreImageUrl,  CurImageUrl, PreImageClips, CurImageClips
# string, string, float, float
# PreImageClips = [coord['左上X'], coord['左上Y'], coord['右下X'], coord['右下Y']]

def _get_data_beta():  # for test only, init function
    url = 'https://infrared.hzncc.cn/api/infrared/alarm/list/byNumber/2'
    req = requests.get(url)
    imgs_data = json.loads(req.content)
    data1, data2 = imgs_data['data'][0:2]
    # data1['可见光图片']

    PreImageUrl = data1['可见光图片']
    CurImageUrl = data2['可见光图片']
    PreImageClips = data1['坐标']
    CurImageClips = data2['坐标']
    return PreImageUrl, CurImageUrl, PreImageClips, CurImageClips


def parse_coord(coords):
    # coords: [{..., "alarmVlPicLeftTopX": 332.0,"alarmVlPicLeftTopY": 274.0,
    #           "alarmVlPicRightBottomX": 444.0,"alarmVlPicRightBottomY": 390.0},{...},...]
    coords_cpy = []
    for coord in coords:
        # coords_cpy.append([coord['左上X'], coord['左上Y'], coord['右下X'], coord['右下Y']])
        coords_cpy.append([coord["alarmVlPicLeftTopX"], coord["alarmVlPicLeftTopY"], coord["alarmVlPicRightBottomX"],
                           coord["alarmVlPicRightBottomY"]])
    return coords_cpy


def get_image_clips(imageUrl, imageClips):
    imageBytes = requests.get(imageUrl).content
    image = cv2.imdecode(np.array(bytearray(imageBytes), np.uint8), -1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    img_clips = []
    for image_coord in imageClips:
        image_coord = map(int, image_coord)
        clip = image[image_coord[1]:image_coord[3], image_coord[0]:image_coord[2]]
        img_clips.append({'clip': clip, 'coord': image_coord})

    image_clips = {'source_img': image, 'img_clips': img_clips}
    return image_clips


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
                            min(y2 + height_expand * 2, raw_height * height / clip_height)]
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
    if np.array(descriptor1 == None).all() or np.array(descriptor2 == None).all(): return 0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.knnMatch(descriptor1, descriptor2, k=1)
    if np.count_nonzero(matches) == 0: return 0

    max_dist = 0
    min_dist = 100
    good_num = 0
    for i in matches:
        min_dist = min(min_dist, i[0].distance) if i != [] else min_dist
        max_dist = max(max_dist, i[0].distance) if i != [] else max_dist
        good_num = good_num + 1 if i != [] and i[0].distance <= Distance_threshold else good_num
    similary = float(good_num) / np.count_nonzero(matches)
    return similary


def HammingDistace(img_1, img_2):  # opencv types
    featureImage = cv2.resize(img_1, (8, 8))
    hashcode1 = (featureImage >= np.mean(np.mean(featureImage))).astype(np.int8)
    featureImage = cv2.resize(img_2, (8, 8))
    hashcode2 = (featureImage >= np.mean(np.mean(featureImage))).astype(np.int8)
    return np.count_nonzero(hashcode1 != hashcode2)


# def CosineDistace(img_1, img_2):
#     pass
#     return cosine_dis


def compare_To(img1, img2):
    orb_similarity = []
    orb_similarity_index = []
    orb_similarity_index_coord = []
    # cosine_distance = []
    # cosine_distance_index = []
    # cosine_distance_index_coord = []
    # hamming_distance = []
    # hamming_distance_index = []

    img1_raw = img1['source_img']
    [raw_height, raw_width] = img1_raw.shape
    for clip1 in img1['img_clips']:
        clip_shape = clip1['clip'].shape
        [height, width] = clip_shape
        [height_expand, width_expand] = map(int, [height / 2, width / 3])
        [x1, y1, x2, y2] = clip1['coord']
        [x1, y1, x2, y2] = [max(0, x1 - width_expand), max(0, y1 - height_expand), min(x2 + width_expand, raw_width),
                            min(y2 + int(height_expand * 1.5), raw_height)]
        clip_source = img1_raw[y1:y2, x1:x2]
        clip_templates = get_expand_clip_templates(clip_shape, img2)

        max_similarity = 0
        min_hamming_dis = 1024
        orb_similarity_temp = []
        # cosine_distace_temp = []
        # hamming_distance_temp = []
        for clip_template in clip_templates:
            orb_similarity_temp.append(OrbSimilarity(clip_source, clip_template))
            # cosine_distace_temp.append(CosineDistace(clip_source, clip_template))
            # hamming_distance_temp.append(HammingDistace(clip_source, clip_template))

        max_orb_similarity = max(orb_similarity_temp)
        max_orb_similarity_index = orb_similarity_temp.index(
            max_orb_similarity) if max_orb_similarity > Orb_threshold else -1
        max_orb_similarity_index_coord = img2['img_clips'][max_orb_similarity_index][
            'coord'] if max_orb_similarity > Orb_threshold else -1
        orb_similarity.append(max_orb_similarity)
        orb_similarity_index.append(max_orb_similarity_index)
        orb_similarity_index_coord.append(max_orb_similarity_index_coord)  # [[x1, y1, x2, y2],-1,[],-1,...]

        # min_cosine_dis = min(cosine_distace_temp)
        # min_cosine_dis_index = cosine_distace_temp.index(min_cosine_dis) if min_cosine_dis < Cosine_threshold else -1
        # min_cosine_dis_index_coord = img2['img_clips'][min_cosine_dis_index][
        #     'coord'] if min_cosine_dis < Cosine_threshold else -1
        # cosine_distance.append(min_cosine_dis)
        # cosine_distance_index.append(min_cosine_dis_index)
        # cosine_distance_index_coord.append(min_cosine_dis_index_coord)  # [[x1, y1, x2, y2],-1,[],-1,...]

        # min_hamming_dis = min(hamming_distance_temp)
        # min_hamming_dis_index = hamming_distance_temp.index(
        #     min_hamming_dis) if min_hamming_dis < Hamming_threshold else -1
        # hamming_distance.append(min_hamming_dis)
        # hamming_distance_index.append(min_hamming_dis_index)

    # if min(orb_similarity) > Orb_threshold:
    #     return True, orb_similarity_index
    # elif max(cosine_distance) < Cosine_threshold:
    #     return True, cosine_distance_index
    # elif max(hamming_distance) < Hamming_threshold:
    #     return True, hamming_distance_index
    # else: return False, orb_similarity_index
    return max(orb_similarity) < Orb_threshold, orb_similarity_index, orb_similarity_index_coord


def is_Duplicated(PreImageUrl, CurImageUrl, PreImageClips, CurImageClips):
    PreImageClips = parse_coord(PreImageClips)
    CurImageClips = parse_coord(CurImageClips)
    img1 = get_image_clips(PreImageUrl, PreImageClips)
    img2 = get_image_clips(CurImageUrl, CurImageClips)
    if len(img1['img_clips']) != len(img2['img_clips']):
        return False, None
    img1_to_img2, img1_to_img2_index, img1_to_img2_index_coord = compare_To(img1, img2)
    img2_to_img1, img2_to_img1_index, img2_to_img1_index_coord = compare_To(img2, img1)

    img1to2index = []
    img1to2index_coord = []
    for i in range(len(img1_to_img2_index)):
        if img1_to_img2_index[i] == -1 or img2_to_img1_index[img1_to_img2_index[i]] != i:
            img1to2index.append(-1)
            # img1to2index_coord.append(img1['img_clips'][i]['coord'])
        else:
            img1to2index.append(img1_to_img2_index[i])
    unique_index = [i for i, x in enumerate(img1to2index) if x == -1]
    unique_img1_coord = [img1['img_clips'][i]['coord'] for i in unique_index]

    img2to1index = []
    img2to1index_coord = []
    for i in range(len(img2_to_img1_index)):
        if img2_to_img1_index[i] == -1 or img1_to_img2_index[img2_to_img1_index[i]] != i:
            img2to1index.append(-1)
            # img2to1index_coord.append(img2['img_clips'][i]['coord'])
        else:
            img2to1index.append(img2_to_img1_index[i])
    unique_index = [i for i, x in enumerate(img2to1index) if x == -1]
    unique_img2_coord = [img2['img_clips'][i]['coord'] for i in unique_index]

    if img1_to_img2 and img2_to_img1:
        return True, unique_img1_coord, unique_img2_coord
    else:
        return False, unique_img1_coord, unique_img2_coord


if __name__ == '__main__':
    PreImageUrl, CurImageUrl, PreImageClips, CurImageClips = _get_data_beta()
    # PreImage = get_image_clips(PreImageUrl, PreImageClips)
    # CurImage = get_image_clips(CurImageUrl, CurImageClips)
    # # imgs = get_image_clips(url)
    # # img1, img2 = imgs[0], imgs[1]
    # cv2.imwrite(os.path.join(os.getcwd(), 'img0.jpg'), PreImage['source_img'])
    # cv2.imwrite(os.path.join(os.getcwd(), 'img1.jpg'), CurImage['source_img'])
    # cv2.imwrite(os.path.join(os.getcwd(), 'img0clip.jpg'), PreImage['img_clips'][0]['clip'])
    # cv2.imwrite(os.path.join(os.getcwd(), 'img1clip.jpg'), CurImage['img_clips'][0]['clip'])
    print(is_Duplicated(PreImageUrl, CurImageUrl, PreImageClips, CurImageClips))
