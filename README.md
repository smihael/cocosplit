Dependencies for both tools can be installed using `pip install -r requirements.txt`.

# LabelMe to COCO converter
Simple tool to:
1) convert labelme annotations to coco format
2) split coco annotations (json) into train and test sets.

Based on:
- https://github.com/akarazniewicz/cocosplit
- https://github.com/wkentaro/labelme/blob/main/examples/instance_segmentation/labelme2coco.py

## Usage

```
$ python labelme2cocosplit.py -h          
usage: labelme2coco.py [-h] [--noviz] [-s SPLIT] [--having-annotations]
                       input_dir output_dir

positional arguments:
  input_dir             input annotated directory
  output_dir            output dataset directory

optional arguments:
  -h, --help            show this help message and exit
  --noviz               no visualization (default: False)
  -s SPLIT              A percentage of a split; a number in (0, 1) (default:
                        0.8)
  --having-annotations  Ignore all images without annotations. Keep only these
                        with at least one annotation (default: False)
```

# Running

```
$ python3 labelme2cocosplit.py --having-annotations /path/to/your/labelme/annotations /output/path
```

will convert labelme annotations to coco format and split it into ``train.json`` and ``test.json`` with ratio 80%/20% respectively. It will skip all
images (``--having-annotations``) without annotations.


# COCO to LabelMe Converter

This script converts annotations from the COCO dataset format to the LabelMe format. It supports both training and testing data splits and allows you to save the annotations in the LabelMe format in the specified directory.

Based on:
- https://gist.github.com/travishsu/6efa5c9fb92ece37b4748036026342f6

## Usage

The script provides a command-line interface. You can run it with the following options:

```
$ python coco2labelme.py  -h
usage: coco2labelme.py [-h] [--train_annotation TRAIN_ANNOTATION] [--test_annotation TEST_ANNOTATION] --images IMAGES --output OUTPUT [--overwrite]

Convert COCO dataset to LabelMe format for train and test splits.

options:
  -h, --help            show this help message and exit
  --train_annotation TRAIN_ANNOTATION
                        Path to the training COCO dataset annotation JSON file.
  --test_annotation TEST_ANNOTATION
                        Path to the testing COCO dataset annotation JSON file.
  --images IMAGES       Path to the COCO images directory.
  --output OUTPUT       Path to the output directory for LabelMe format.
  --overwrite           Overwrite existing files in the output directory.
```

# Running

```bash
python coco2labelme.py --train_annotation <path_to_train_json> --test_annotation <path_to_test_json> --images <path_to_images> --output <output_dir> [--overwrite]
```
