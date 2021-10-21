#!/usr/bin/env python3

import argparse
import collections
import datetime
import glob
import json
import os
import os.path as osp
import sys
import uuid
import funcy
from sklearn.model_selection import train_test_split

import imgviz
import numpy as np

import labelme

try:
    import pycocotools.mask
except ImportError:
    print("Please install pycocotools:\n\n    pip install pycocotools\n")
    sys.exit(1)

def save_coco(file, info, licenses, images, annotations, categories):
    with open(file, 'wt', encoding='UTF-8') as coco:
        json.dump({ 'info': info, 'licenses': licenses, 'images': images, 
            'annotations': annotations, 'categories': categories}, coco, indent=2, sort_keys=True)

def filter_annotations(annotations, images):
    image_ids = funcy.lmap(lambda i: int(i['id']), images)
    return funcy.lfilter(lambda a: int(a['image_id']) in image_ids, annotations)
   
    
def class_name_to_id(label,categories):
    for category in categories:
        if label == category['name']:
            return category['id']
    return -1
                
def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("input_dir", help="input annotated directory")
    parser.add_argument("output_dir", help="output dataset directory")
    #parser.add_argument("--labels", help="labels file", required=True)
    parser.add_argument(
        "--noviz", help="no visualization", action="store_true"
    )
    #parser.add_argument('annotations', metavar='coco_annotations', type=str,
    #                help='Path to COCO annotations file.')
    #parser.add_argument('train', type=str, help='Where to store COCO training annotations')
    #parser.add_argument('test', type=str, help='Where to store COCO test annotations')
    parser.add_argument('-s', dest='split', type=float, default=0.8,
                        help="A percentage of a split; a number in (0, 1)")
    parser.add_argument('--having-annotations', dest='having_annotations', action='store_true',
                        help='Ignore all images without annotations. Keep only these with at least one annotation')
    args = parser.parse_args()

    if osp.exists(args.output_dir):
        print("Output directory already exists:", args.output_dir)
        sys.exit(1)
    os.makedirs(args.output_dir)
    os.makedirs(osp.join(args.output_dir, "JPEGImages"))
    if not args.noviz:
        os.makedirs(osp.join(args.output_dir, "Visualization"))
    print("Creating dataset:", args.output_dir)

    now = datetime.datetime.now()

    data = dict(
        info=dict(
            description=None,
            url=None,
            version=None,
            year=now.year,
            contributor=None,
            date_created=now.strftime("%Y-%m-%d %H:%M:%S.%f"),
        ),
        licenses=[dict(url=None, id=0, name=None,)],
        images=[
            # license, url, file_name, height, width, date_captured, id
        ],
        type="instances",
        annotations=[
            # segmentation, area, iscrowd, image_id, bbox, category_id, id
        ],
        categories=[
            # supercategory, id, name
        ],
    )

    #class_name_to_id = {}
    #for i, line in enumerate(open(args.labels).readlines()):
    #    class_id = i - 1  # starts with -1
    #    class_name = line.strip()
    #    if class_id == -1:
    #        assert class_name == "__ignore__"
    #        continue
    #    class_name_to_id[class_name] = class_id
    #    data["categories"].append(
    #        dict(supercategory=None, id=class_id, name=class_name,)
    #    )
    data["categories"].append(dict(supercategory=None, id=0, name="_background_"))
    seen_labels = list() # for seen labels

    out_ann_file = osp.join(args.output_dir, "annotations.json")
    label_files = glob.glob(osp.join(args.input_dir, "*.json"))
    for image_id, filename in enumerate(label_files):
        #print("Generating dataset from:", filename)

        label_file = labelme.LabelFile(filename=filename)

        base = osp.splitext(osp.basename(filename))[0]
        out_img_file = osp.join(args.output_dir, "JPEGImages", base + ".jpg")

        img = labelme.utils.img_data_to_arr(label_file.imageData)
        imgviz.io.imsave(out_img_file, img)
        data["images"].append(
            dict(
                license=0,
                url=None,
                file_name=osp.relpath(out_img_file, osp.dirname(out_ann_file)),
                height=img.shape[0],
                width=img.shape[1],
                date_captured=None,
                id=image_id,
            )
        )

        masks = {}  # for area
        segmentations = collections.defaultdict(list)  # for segmentation
        
        for shape in label_file.shapes:
            points = shape["points"]
            label = shape["label"]
            group_id = shape.get("group_id")
            shape_type = shape.get("shape_type", "polygon")
            mask = labelme.utils.shape_to_mask(
                img.shape[:2], points, shape_type
            )

            if group_id is None:
                group_id = uuid.uuid1()

            instance = (label, group_id)

            if instance in masks:
                masks[instance] = masks[instance] | mask
            else:
                masks[instance] = mask

            if shape_type == "rectangle":
                (x1, y1), (x2, y2) = points
                x1, x2 = sorted([x1, x2])
                y1, y2 = sorted([y1, y2])
                points = [x1, y1, x2, y1, x2, y2, x1, y2]
            else:
                points = np.asarray(points).flatten().tolist()

            segmentations[instance].append(points)
        segmentations = dict(segmentations)

        for instance, mask in masks.items():
            cls_name, group_id = instance
            
            #import pdb; pdb.set_trace()

            #if cls_name not in class_name_to_id:
            #    continue
            #cls_id = class_name_to_id[cls_name]
            if cls_name not in seen_labels and cls_name != "__ignore__":
                seen_labels.append(cls_name)
                data["categories"].append(dict(supercategory=None, id=len(seen_labels), name=cls_name))                
            cls_id = class_name_to_id(cls_name,data["categories"])

            mask = np.asfortranarray(mask.astype(np.uint8))
            mask = pycocotools.mask.encode(mask)
            area = float(pycocotools.mask.area(mask))
            bbox = pycocotools.mask.toBbox(mask).flatten().tolist()

            data["annotations"].append(
                dict(
                    id=len(data["annotations"]),
                    image_id=image_id,
                    category_id=cls_id,
                    segmentation=segmentations[instance],
                    area=area,
                    bbox=bbox,
                    iscrowd=0,
                )
            )

        cat_ids = {}
        for category in data["categories"]:
            cat_ids[category["name"]] = category["id"]
            
        if not args.noviz:
            viz = img
            if masks:
                labels, captions, masks = zip(
                    *[
                        #(class_name_to_id(cnm,data["categories"]), cnm, msk)
                        #(class_name_to_id[cnm], cnm, msk)
                        (cat_ids[cnm], cnm, msk)
                        for (cnm, gid), msk in masks.items()
                        #if cnm in class_name_to_id
                        if cnm in cat_ids
                    ]
                )
                
                viz = imgviz.instances2rgb(
                    image=img,
                    labels=labels,
                    masks=masks,
                    captions=captions,
                    font_size=15,
                    line_width=2,
                )
            out_viz_file = osp.join(
                args.output_dir, "Visualization", base + ".jpg"
            )
            imgviz.io.imsave(out_viz_file, viz)

    #with open(out_ann_file, "w") as f:
    #    json.dump(data, f)
    
    info = data['info']
    licenses = data['licenses']
    images = data['images']
    annotations = data['annotations']
    categories = data['categories']

    number_of_images = len(images)

    images_with_annotations = funcy.lmap(lambda a: int(a['image_id']), annotations)

    if args.having_annotations:
        images = funcy.lremove(lambda i: i['id'] not in images_with_annotations, images)

    x, y = train_test_split(images, train_size=args.split)

    train_file=osp.join(args.output_dir,'train.json')
    test_file=osp.join(args.output_dir,'test.json')
    save_coco(train_file, info, licenses, x, filter_annotations(annotations, x), categories)
    save_coco(test_file, info, licenses, y, filter_annotations(annotations, y), categories)

    print("Saved {} entries in {} and {} in {}".format(len(x), train_file, len(y), test_file))



if __name__ == "__main__":
    main()
