import glob
import os
import json

import shapely

driveltr = "d:"

rawdir = os.path.join(driveltr, r"datasets\PV Aerial\NY\Raw\boro_queens_sp18")
maskdir = os.path.join(driveltr, r"datasets\PV Aerial\NY\mask")
jsondir = os.path.join(driveltr, r"datasets\PV Aerial\NY\json")


def print_maybes():
    fns = glob.glob(os.path.join(jsondir, "*.json"))

    for fn in fns:
        with open(fn) as f:
            if "maybe" in f.read():
                print(os.path.basename(fn))

def notpv_within_pv():
    fns = glob.glob(os.path.join(jsondir, "*.json"))

    n_not = 0
    fails = 0

    for fn in fns:
        with open(fn) as f:
            data = json.load(f)

            shapes = data["shapes"]
            pv = [shape for shape in shapes if shape["label"] == "pv"]
            notpv = [shape for shape in shapes if shape["label"] == "notpv"]



            for notpvshape in notpv:
                confirmed = False
                # Check if it is within a pv shape
                notpv_shapely = shapely.geometry.Polygon(notpvshape["points"])
                for pvshape in pv:
                    pv_shapely = shapely.geometry.Polygon(pvshape["points"])
                    if notpv_shapely.within(pv_shapely):
                        confirmed = True
                        n_not += 1
                        break
                if not confirmed:
                    fails += 1
                    print(os.path.basename(fn))

    print(f"{n_not} notpv shapes within pv shapes")
    print(f"{fails} notpv shapes failed test")

if __name__ == "__main__":
    # print_maybes()
    notpv_within_pv()