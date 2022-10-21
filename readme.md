# Data Source

New York Digital Orthodata
[Link](https://gis.ny.gov/gateway/mg/)

# Setup Instructions
Install [python](https://www.python.org)

### Install packages
Open a command prompt and type the following
```
pip install pillow
pip install numpy
```

# Process Data
Download a file. Edit `convert_to_png.py` to point to the correct file

Original Data File: Queens - State Plane 2018
[Direct Download](https://gis.ny.gov/gateway/mg/2018/new_york_city/)

# Labelling
Install [LabelMe](https://github.com/wkentaro/labelme).

I used the prebuilt binaries but you could follow different installation
instructions.

Within LabelMe, open the folder where all the PNG files have been saved. 
Create polygon labels (you will need to zoom in). The polygons should closely
hug the PV arrays and should only cover the actual PV panels themselves. Here's 
an example:
![Sample Image](example_label.png "Rooftop")

I believe we should get started with two separate labels in the dataset:
- `pv` for obvious pv installations
- `notpv` for empty rings within a larger polygon. May need to revisit this.
- `maybe` for uncertain spots that we can review.

Some detailed LabelMe examples [here](https://datagen.tech/guides/image-annotation/labelme/).

# Workflow
### Download data
### Convert JPEG2000 to PNG
- `convert_to_png.py` is a script to convert a whole directory of files into png format
### Label the Data
- Use `labelme`. Optionally, use `remove_json_imagedata.py` to clear the imagedata from saved json files
- Use `generate_blank_json.py` to fill in any JSON files for images that have no objects
### Convert JSON Polygons to Mask
- Use `json_to_dataset.py` to convert a directory of the JSON files into binary mask images
### Slice the Images to Appropriate Size
- Use `slice_dataset_tiles.py` to slice the images into proper sizes that will be used by the CNN


# Labelled Neural Net Datasets
## New York
Created here from [New York Orthodata](https://gis.ny.gov/gateway/mg/)
## France
From public [Zenodo Repository](https://zenodo.org/record/7059985)
## California
Originating at [Figshare Repository](https://figshare.com/articles/dataset/Distributed_Solar_Photovoltaic_Array_Location_and_Extent_Data_Set_for_Remote_Sensing_Object_Identification/3385780/1?file=5286613)
