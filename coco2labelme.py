import os
import json
import argparse
import numpy as np
import pandas as pd
from skimage.measure import find_contours
#import base64
import shutil  


class CocoDatasetHandler:
    def __init__(self, jsonpath, imgpath):
        """
        Initialize the handler by loading COCO annotations and images.
        jsonpath: Path to the COCO annotations JSON file.
        imgpath: Path to the directory containing the image files.
        """
        with open(jsonpath, 'r') as jsonfile:
            ann = json.load(jsonfile)

        # Load images, annotations, and categories as pandas DataFrames
        images = pd.DataFrame.from_dict(ann['images']).set_index('id')
        annotations = pd.DataFrame.from_dict(ann['annotations']).set_index('id')
        categories = pd.DataFrame.from_dict(ann['categories']).set_index('id')

        # Merge annotations with images and categories to include relevant metadata
        annotations = annotations.merge(images, left_on='image_id', right_index=True)
        annotations = annotations.merge(categories, left_on='category_id', right_index=True)

        # Convert COCO annotations to LabelMe format shapes
        annotations = annotations.assign(
            shapes=annotations.apply(self.coco2shape, axis=1)
        )
        self.annotations = annotations
        self.labelme = {}

        self.imgpath = imgpath
        self.images = pd.DataFrame.from_dict(ann['images']).set_index('file_name')

    def coco2shape(self, row):
        """
        Convert COCO annotation to LabelMe-compatible shapes.
        """
        if row.iscrowd == 1:
            shapes = self.rle2shape(row)
        elif row.iscrowd == 0:
            shapes = self.polygon2shape(row)
        return shapes

    def rle2shape(self, row):
        """
        Convert run-length encoding (RLE) segmentation to polygons.
        """
        rle, shape = row['segmentation']['counts'], row['segmentation']['size']
        mask = self._rle_decode(rle, shape)
        padded_mask = np.zeros((mask.shape[0]+2, mask.shape[1]+2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        points = find_contours(mask, 0.5)
        shapes = [[[int(point[1]), int(point[0])] for point in polygon] for polygon in points]
        return shapes

    def _rle_decode(self, rle, shape):
        """
        Decode the RLE-encoded mask into a binary mask.
        """
        mask = np.zeros([shape[0] * shape[1]], np.bool)
        for idx, r in enumerate(rle):
            if idx < 1:
                s = 0
            else:
                s = sum(rle[:idx])
            e = s + r
            if e == s:
                continue
            assert 0 <= s < mask.shape[0]
            assert 1 <= e <= mask.shape[0]
            if idx % 2 == 1:
                mask[s:e] = 1
        mask = mask.reshape([shape[1], shape[0]]).T
        return mask

    def polygon2shape(self, row):
        """
        Convert polygon segmentation to LabelMe-compatible shape format.
        """
        shapes = [[[int(points[2*i]), int(points[2*i+1])] for i in range(len(points)//2)] for points in row.segmentation]
        return shapes

    def coco2labelme(self):
        """
        Convert COCO annotations to LabelMe format.
        """
        fillColor = [255, 0, 0, 128]
        lineColor = [0, 255, 0, 128]

        groups = self.annotations.groupby('file_name')
        
        for filename, df in groups:
            # Load and base64 encode the image file
            
            #img_path = os.path.join(self.imgpath, filename)
            #with open(img_path, 'rb') as img_file:
            #    image_data = base64.b64encode(img_file.read()).decode('utf-8')

            record = {
                #'imageData': image_data,  # Base64-encoded image data
                'imageData': None,
                'fillColor': fillColor,
                'lineColor': lineColor,
                'imagePath': os.path.basename(filename), 
                #'imagePath': filename, 
                'imageHeight': int(self.images.loc[filename].height),
                'imageWidth': int(self.images.loc[filename].width),
                'shapes': []
            }

            instance = {'line_color': None, 'fill_color': None, 'shape_type': "polygon"}
            for inst_idx, (_, row) in enumerate(df.iterrows()):
                for polygon in row.shapes:
                    copy_instance = instance.copy()
                    copy_instance.update({
                        'label': row['name'],
                        'group_id': inst_idx,
                        'points': polygon
                    })
                    record['shapes'].append(copy_instance)

            # Store the labelme annotation for the current image
            self.labelme[filename] = record


    def save_labelme(self, file_names, dirpath, save_json_only=False, overwrite=False):
        """
        Save the annotations in LabelMe format and copy the corresponding images to the output directory.

        Args:
            file_names (iterable): The file names for the annotations (including extensions like .jpg).
            dirpath (str): Directory where LabelMe annotations and images will be saved.
            save_json_only (bool): If True, only save the JSON annotations.
            overwrite (bool): If True, overwrite individual files if they exist.
        """
        # Ensure the base directory exists
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)

        # Saving logic
        for file_name in file_names:

            # Get the base name without the directory
            base_name_with_extension = os.path.basename(file_name)

            # Remove the extension
            base_name_without_extension = os.path.splitext(base_name_with_extension)[0]

            # Ensure subdirectory exists (if the file is in a subdirectory)
            annotation_file = os.path.join(dirpath, f"{base_name_without_extension}.json")
            subdir = os.path.dirname(annotation_file)
            if not os.path.exists(subdir):
                os.makedirs(subdir)

            # If the file already exists and overwrite is False, skip saving it
            if os.path.exists(annotation_file) and not overwrite:
                print(f"File {annotation_file} already exists, skipping.")
                continue

            # Prepare the annotation data
            labelme_annotation = self.labelme[file_name]

            # Save the annotation to the JSON file
            with open(annotation_file, 'w') as f:
                json.dump(labelme_annotation, f, indent=4)

            if not save_json_only:
                # Copy the corresponding image to the output directory
                image_src_path = os.path.join(self.imgpath, file_name)
                image_dest_path = os.path.join(dirpath, base_name_with_extension)

                # Make sure the source image exists before copying
                if os.path.exists(image_src_path):
                    # Ensure destination subdirectories exist
                    if not os.path.exists(os.path.dirname(image_dest_path)):
                        os.makedirs(os.path.dirname(image_dest_path))

                    shutil.copy(image_src_path, image_dest_path)
                    print(f"Copied image {image_src_path} to {image_dest_path}")
                else:
                    print(f"Image {image_src_path} not found, skipping.")

        print(f"Annotations and images saved to {dirpath}")

def process_dataset(annotation_path, image_dir, output_dir, overwrite=False):
    """
    Process the COCO dataset and convert it to LabelMe format.
    """
    # Create a CocoDatasetHandler object
    ds = CocoDatasetHandler(annotation_path, image_dir)
    
    # Convert COCO annotations to LabelMe format
    ds.coco2labelme()
    
    # Save the converted annotations
    ds.save_labelme(ds.labelme.keys(), output_dir, overwrite=overwrite)

if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description="Convert COCO dataset to LabelMe format for train and test splits.")

    # Add arguments for train and test annotation paths
    parser.add_argument('--train_annotation', type=str, help="Path to the training COCO dataset annotation JSON file.")
    parser.add_argument('--test_annotation', type=str, help="Path to the testing COCO dataset annotation JSON file.")
    parser.add_argument('--images', type=str, required=True, help="Path to the COCO images directory.")
    parser.add_argument('--output', type=str, required=True, help="Path to the output directory for LabelMe format.")
    parser.add_argument('--overwrite', action='store_true', help="Overwrite existing files in the output directory.")
    
    # Parse the command-line arguments
    args = parser.parse_args()
    
    # Process training dataset if provided
    if args.train_annotation:
        print(f"Processing training data from {args.train_annotation}")
        process_dataset(args.train_annotation, args.images, args.output, args.overwrite)
    
    # Process testing dataset if provided
    if args.test_annotation:
        print(f"Processing testing data from {args.test_annotation}")
        process_dataset(args.test_annotation, args.images, args.output, args.overwrite)
