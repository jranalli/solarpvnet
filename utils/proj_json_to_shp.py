import os
import glob
import json
import shapely
import tqdm
import pandas as pd
import geopandas


def get_wkt():
    """
    Return the WKT representation of the map projection. The WKT can be found in
    the .prj file that accompanies the downloaded Queens dataset, specifically
    "lot18_nyc_liz_06.prj"
    """

    wkt = '''PROJCS["NAD_1983_StatePlane_New_York_Long_Island_FIPS_3104_Feet",GEOGCS["GCS_North_American_1983",DATUM["D_North_American_1983",SPHEROID["GRS_1980",6378137.0,298.257222101]],PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]],PROJECTION["Lambert_Conformal_Conic"],PARAMETER["False_Easting",984250.0],PARAMETER["False_Northing",0.0],PARAMETER["Central_Meridian",-74.0],PARAMETER["Standard_Parallel_1",40.66666666666666],PARAMETER["Standard_Parallel_2",41.03333333333333],PARAMETER["Latitude_Of_Origin",40.16666666666666],UNIT["Foot_US",0.3048006096012192]]'''

    return wkt

def read_tile_table(datasetdir):
    """
    Read the tile table from the .dbf file showing the tile extents in pixels.
    The relevant file is "18ic_b_queens_l06_4bd.dbf" included in the Queens
    dataset.
    """

    with open(os.path.join(datasetdir, "18ic_b_queens_l06_4bd.dbf"), "r") as file:
        _ = next(file)
        # Data exists in second line of file, formatted as fixed width values
        data = next(file)[1:-1]

        # Split it into individual entries
        data = data.split()

        # The names for the columns
        columns = ["tile", "xmin", "ymin", "xmax", "ymax"]

        # There are 5 entries for each tile all arranged in a single text line.
        # Split into multiple rows
        rows = [data[i:i+5] for i in range(0, len(data), 5)]
        rows.sort()  # automatically sorts by first element in sublist

        # Convert to DF for easier use
        df = pd.DataFrame(rows, columns=columns)

    return df

def create_shapefile(jsondir, tilepositions, shp_outdir="shapefile"):
    """
    Load the labelme json files and convert to a shapefile.
    tilepositions is the information for each tile obtained from read_tile_table()
    """


    plot = False

    fns = glob.glob(os.path.join(jsondir, "*.json"))

    # A list of shapely polygons in all images
    all_polygons = []

    for fn in tqdm.tqdm(fns):

        with open(fn, "r") as file:

            data = json.load(file)
            shapes = data["shapes"]

            # Separate pv from notpv
            pv = [shape for shape in shapes if shape["label"] == "pv"]
            notpv = [shape for shape in shapes if shape["label"] == "notpv"]

            # We need to keep track of shapes that contain inner 'notpv' rings

            # Dict that will be keyed by shapely pv object and value will be a list of the notpv json objects it contains
            pv_with_inner = {}
            # List of all the pv json objects that contain inner notpv rings
            pv_with_inner_json = []

            total_pv_count = len(pv)

            # Find out which pv shape each notpv corresponds to
            for notpvshape in notpv:

                # create a shapely object version of the geometry
                notpv_shapely = shapely.geometry.Polygon(notpvshape["points"])

                for pvshape in pv:

                    # Create a shapely object version of the geometry
                    pv_shapely = shapely.geometry.Polygon(pvshape["points"])

                    # Test if the notpv shape is contained
                    if notpv_shapely.within(pv_shapely):
                        # Initialize the dict if this pv has not yet had a notpv inside, otherwise just add it
                        if pv_shapely not in pv_with_inner:
                            pv_with_inner[pv_shapely] = [notpvshape]
                            pv_with_inner_json.append(pvshape)
                        else:
                            pv_with_inner[pv_shapely].append(notpvshape)
                        break

        # Check that we have accounted for all the notpv
        assert (sum([len(v) for v in pv_with_inner.values()]) == len(notpv))

        # We now have a list of all the pvjson stored in pv. It's time to build shapely polygons for each, nesting the notpv if needed
        # All the pv that contain a notpv are stored by json in pv_with_inner_json, and keyed by shapely version of pv in pv_with_inner

        # A list of shapely polygons in this image
        this_polygons = []

        # Get the x & y positions of the top left of each image in the projected coordinates
        x_tl = int(tilepositions.loc[tilepositions['tile']==os.path.basename(fn).split(".")[0]+".jp2"]['xmin'].values[0])
        y_tl = int(tilepositions.loc[tilepositions['tile']==os.path.basename(fn).split(".")[0]+".jp2"]['ymax'].values[0])
        # A transform for defining pixel coordinates to projection coordinates
        # Resolution is 0.5 x 0.5 feet per pixel, negative sign accounts for top left origin.
        affmat = [0.5, 0, 0, -0.5, x_tl, y_tl]

        # Account for all PV that has an inner notpv hole
        for pvjson in pv_with_inner_json:
            pv_shapely_ext = shapely.geometry.Polygon(pvjson["points"])  # shapely copy for keying the dict
            holes = [notpv_i['points'] for notpv_i in pv_with_inner[pv_shapely_ext]]

            # Create the polygon with hols, and transform it
            poly = shapely.geometry.Polygon(pvjson['points'], holes=holes)
            poly = shapely.affinity.affine_transform(poly, affmat)

            # Add it to the overall list
            this_polygons.append(poly)

            # Remove from the larger list of pv
            pv.pop(pv.index(pvjson))

        # Now create the polygons for the remaining pv that do not have holes
        for pvjson in pv:
            poly = shapely.geometry.Polygon(pvjson['points'])
            poly = shapely.affinity.affine_transform(poly, affmat)
            this_polygons.append(poly)

        # Check that we have correctly kept track of all the PV
        assert(len(this_polygons) == total_pv_count)

        # Add to the global list of polygons
        [all_polygons.append(polygon) for polygon in this_polygons]

        # If desired, plot it to make sure we did everything right
        if plot:
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

    # Create and save a geopandas dataframe
    gdf = geopandas.GeoDataFrame(geometry=all_polygons, crs=get_wkt())
    gdf.to_file(shp_outdir)

        

if __name__ == "__main__":
    driveltr = "d:"

    myrawdir = os.path.join(driveltr,
                            r"datasets\PV Aerial\NY\Raw\boro_queens_sp18")
    mymaskdir = os.path.join(driveltr, r"datasets\PV Aerial\NY\mask")
    myjsondir = os.path.join(driveltr, r"datasets\PV Aerial\NY\json")

    table = read_tile_table(myrawdir)
    create_shapefile(myjsondir, table)