# Data Source

New York Digital Orthodata
[Link](http://gis.ny.gov/gateway/mg/)

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
[Direct Download](http://gis.ny.gov/gateway/mg/2018/new_york_city/)

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