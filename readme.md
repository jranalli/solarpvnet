# SolarPVNet
SolarPVNet is a project to create a neural network that can identify solar PV 
panels from aerial imagery. The repository contains various tools and codes
that have been used to generate data in research articles published by the 
author.

***************
## Data Sources
Data from the following six datasets has been incorporated into existing and  
in-progress publications. All datasets require some preprocessing to configure 
for the data pipeline used by the additional tools. These can be found in the 
`preprocess` package. Preprocess scripts do not use a nicely unified format due
to the differences between the datasets and will need some individual 
inspection for how to run them.

#### CA-F and CA-S - California, USA
These datasets represent the Fresno and Stockton portions of a dataset using 
USGS orthoimagery from California produced by Bradbury, et al. The dataset
additionally contains labels for Modesto and Oxnard, but these were too small 
for use in the current project (which required 1000+ tiles per site). Images 
are large, and preprocessing is required. See `preprocess_cal.py`.
- [Article](https://www.nature.com/articles/sdata2016106)
- [Data](https://dx.doi.org/10.6084/m9.figshare.3385780)

#### FR-G and FR-I - France
Two datasets by Kasmi, et al. FR-G utilizes data from Google earth, while FR-I
uses data from the French national institute of geographical and forestry 
information (IGN). See `preprocess_france.py`.
- [Article](https://doi.org/10.1038/s41597-023-01951-4)
- [Data](https://doi.org/10.5281/zenodo.7059985)

#### DE-G - Oldenburg, Germany
This dataset was taken from Google Earth imagery of Oldenburg, Germany. The 
data is not publicly available at present. An initial publication utilizing 
the data can be found in the proceedings of the IEEE PVSC. 
- [Article](https://doi.org/10.1109/PVSC45281.2020.9300636)

#### NY-Q - Queens, New York, USA
This dataset consists of all aerial orthoimagery from Queens, New York, for 
the year 2018. Labeling of the dataset is ongoing, but a public version of the
labels is expected to be released in the future. Images are large, and 
preprocessing is required. See `preprocess_nyc.py`.
Available from: New York Digital Orthodata
- [Data Landing Page](https://data.gis.ny.gov/) 
- [Orthodata Direct Link](https://orthos.dhses.ny.gov/)

#### Links to other datasets not used at present
- SolarDK
  - [Article](https://arxiv.org/abs/2212.01260)
  - [Data](https://osf.io/aj539/)

*********************
## Setup Instructions
These tools are written in [python](https://www.python.org) using `tensorflow`,

See `dependencies.txt`

### Install packages
Open a command prompt and type the following
```
pip install pillow
pip install numpy
```

******************
## Processing Data

**********************
### Labelling workflow
The workflow for labelling the NYC data is described here.
- Download data
- Convert JPEG2000 to PNG
  - `convert_to_png.py` contains a function to convert a whole directory of files into png format, stripping the alpha channel
- Label the Data
  - Use `labelme`. Optionally, use `remove_json_imagedata.py` to clear the imagedata accidentally saved to json files.
  - Use `generate_blank_json.py` to fill in any JSON files for images that have no label objects
- Convert JSON Polygons to Mask
  - Use `json_to_dataset.py` to convert a directory of the JSON files into binary mask images
- Slice the Images to Appropriate Size
  - Use `split_image.py` to slice the images into proper sizes that will be used by the CNN

### Installing Labelme
Info found here: [LabelMe](https://github.com/wkentaro/labelme).
I used the prebuilt binaries but you could follow different installation
instructions.

Within LabelMe, open the folder where all the PNG files have been saved. 
Create polygon labels (you will need to zoom in). The polygons should closely
hug the PV arrays and should only cover the actual PV panels themselves. Here's 
an example:
![Sample Image](example_label.png "Rooftop")

Labels used are:
- `pv` for obvious pv installations.
- `notpv` for empty rings within a larger polygon.
- `maybe` for uncertain spots that need review.

Some detailed LabelMe examples [here](https://datagen.tech/guides/image-annotation/labelme/). 