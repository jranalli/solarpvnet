

# Conversation about importing these filetypes to QGIS
# https://discourse.osgeo.org/t/qgis-us-user-newbie-question-about-importing-shapefiles-and-layers-from-arcmaps/110610


# World Files - https://en.wikipedia.org/wiki/World_file
# Lines 5 & 6 are the coordinates in the coordinate system of the top left corner pixel of the image.

# GDAL Transform? - https://gdal.org/programs/gdaltransform.html

# The centers of pixels is offset by half a pixel in both the x and y directions. So that means that the top
# left pixel having a position of 1000000.25 actually has the pixel left at 1000000.0 and the pixel with the
# top left at 139999.75 has the pixel top at 140000.0.

# All data bounds are contained in 18ic_b_queens_l06_4bd.dbf, based on the outer edges, rather than the pixel centers.



# projection name: "NAD_1983_2011_StatePlane_New_York_Long_Isl_FIPS_3104_Ft_US"
# ESRI:102718 - https://epsg.io/102718 - deprecated
# EPSG 9003

# Metadata for the imagery (CCBY 4.0):
# https://github.com/CityOfNewYork/nyc-geo-metadata/blob/main/Metadata/Metadata_AerialImagery.md


# Projection File: lot18_nyc_liz_06.prj
# wkt = '''PROJCS["NAD_1983_StatePlane_New_York_Long_Island_FIPS_3104_Feet",GEOGCS["GCS_North_American_1983",DATUM["D_North_American_1983",SPHEROID["GRS_1980",6378137.0,298.257222101]],PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]],PROJECTION["Lambert_Conformal_Conic"],PARAMETER["False_Easting",984250.0],PARAMETER["False_Northing",0.0],PARAMETER["Central_Meridian",-74.0],PARAMETER["Standard_Parallel_1",40.66666666666666],PARAMETER["Standard_Parallel_2",41.03333333333333],PARAMETER["Latitude_Of_Origin",40.16666666666666],UNIT["Foot_US",0.3048006096012192]]'''

import os
import glob
import pandas as pd

driveltr = "d:"

rawdir = os.path.join(driveltr, r"datasets\PV Aerial\NY\Raw\boro_queens_sp18")
maskdir = os.path.join(driveltr, r"datasets\PV Aerial\NY\mask")
jsondir = os.path.join(driveltr, r"datasets\PV Aerial\NY\json")

def get_wkt():
    """
    Read the projection from the .prj file
    """
    wkt_file = os.path.join(rawdir, "lot18_nyc_liz_06.prj")
    with open(wkt_file, "r") as file:
        wkt = file.read()
    return wkt


def assert_projection():
    """
    Confirm that the projection in each file is the same
    :return:
    AssertionError if not true
    """
    fns = glob.glob(os.path.join(rawdir, "*.jp2"))
    for fn in fns:
        with open(fn, "r", encoding='latin-1') as file:
            # read first line of file
            _ = next(file)
            _ = next(file)
            data = next(file)
            assert 'NAD_1983_2011_StatePlane_New_York_Long_Isl_FIPS_3104_Ft_US' in data


def create_masks():
    from json_to_dataset import labelme_json_to_binary
    label_id_dict = {"_background_": 0, "notpv": 0, "pv": 255}
    layer_order = ["pv", "notpv"]

    for fn in glob.glob(os.path.join(jsondir, "*.json")):
        labelme_json_to_binary(fn, maskdir, label_id_dict, layer_order=layer_order)


def read_tile_table():
    """
    Read the tile table from the dbf file showing the tile extents in pixels
    """
    with open(os.path.join(rawdir, "18ic_b_queens_l06_4bd.dbf"), "r") as file:
        _ = next(file)
        # Data exists in second line of file
        data = next(file)[1:-1]

        # Split it into entries
        data = data.split()

        columns = ["tile", "xmin", "ymin", "xmax", "ymax"]

        # There are 5 entries for each tile, so split into rows
        rows = [data[i:i+5] for i in range(0, len(data), 5)]
        rows.sort()  # automatically by first element in sublist

        # Convert to DF
        df = pd.DataFrame(rows, columns=columns)

    return df

def load_labelme_json():
    """
    Load the labelme json files
    """
    import json
    import shapely

    # fns = glob.glob(os.path.join(jsondir, "*.json"))
    fns = [os.path.join(jsondir, "000207.json")]

    translate_table = read_tile_table()

    for fn in fns:

        pv_with_inner = {}
        pv_with_inner_json = []

        with open(fn, "r") as file:

            data = json.load(file)
            shapes = data["shapes"]
            pv = [shape for shape in shapes if shape["label"] == "pv"]
            notpv = [shape for shape in shapes if shape["label"] == "notpv"]

            total_pv_count = len(pv)

            # Loop over the notpv
            for notpvshape in notpv:

                # create a shapely version
                notpv_shapely = shapely.geometry.Polygon(notpvshape["points"])

                # Loop over the pv to see which contains it
                for pvshape in pv:

                    # Create a shapely version
                    pv_shapely = shapely.geometry.Polygon(pvshape["points"])

                    # Is this the PV that contains it?
                    if notpv_shapely.within(pv_shapely):

                        # Initialize if we haven't stored it
                        if pv_shapely not in pv_with_inner:
                            pv_with_inner[pv_shapely] = [notpvshape]
                            pv_with_inner_json.append(pvshape)
                        else:
                            pv_with_inner[pv_shapely].append(notpvshape)
                        break

        assert (sum([len(v) for v in pv_with_inner.values()]) == len(notpv))
        # print(len(notpv))
        # print(len(pv_with_inner))
        # print([len(v) for v in pv_with_inner.values()])

        # We now have a list of all the pvjson stored in pv.
        # All the pv that contain a notpv are stored by json in pv_with_inner_json, and keyed by shapely version of pv in pv_with_inner

        this_polygons = []

        x_tl = int(translate_table.loc[translate_table['tile']==os.path.basename(fn).split(".")[0]+".jp2"]['xmin'].values[0])
        y_tl = int(translate_table.loc[translate_table['tile']==os.path.basename(fn).split(".")[0]+".jp2"]['ymax'].values[0])
        affmat = [0.5, 0, 0, -0.5, x_tl, y_tl]

        # First account for the PV
        for pvjson in pv_with_inner_json:
            pv_shapely_ext = shapely.geometry.Polygon(pvjson["points"])
            holes = [notpv_i['points'] for notpv_i in pv_with_inner[pv_shapely_ext]]

            poly = shapely.geometry.Polygon(pvjson['points'], holes=holes)
            poly = shapely.affinity.affine_transform(poly, affmat)
            this_polygons.append(poly)

            # Remove this one from the list of PV
            pv.pop(pv.index(pvjson))


        # Now account for the remaining PV
        for pvjson in pv:
            poly = shapely.geometry.Polygon(pvjson['points'])
            poly = shapely.affinity.affine_transform(poly, affmat)
            this_polygons.append(poly)


        assert(len(this_polygons) == total_pv_count)

        # Plot it to verify we did right
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()

        for poly in this_polygons:
            # Plot Polygon
            xe, ye = poly.exterior.xy

            # If there are any Interiors
            # Retrieve coordinates for all interiors
            for inner in poly.interiors:
                xi, yi = zip(*inner.coords[:])
                ax.plot(xi, yi, color="blue")

            ax.plot(xe, ye, color="blue")
        plt.axis('equal')
        plt.show()

        import geopandas
        # Create a GeoDataFrame
        gdf = geopandas.GeoDataFrame(geometry=this_polygons, crs=get_wkt())

        print(gdf)

        

if __name__ == "__main__":

    # assert_projection()
    # read_tile_table()
    # create_masks()
    load_labelme_json()