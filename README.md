Simple tool to:
1) convert labelme annotations to coco format
2) split coco annotations (json) into train and test sets.

Based on:
- https://github.com/akarazniewicz/cocosplit
- https://github.com/wkentaro/labelme/blob/main/examples/instance_segmentation/labelme2coco.py

## Installation

``labelme2cocosplit`` requires python 3 and basic set of dependencies:

```
pip install -r requirements.txt
```

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
