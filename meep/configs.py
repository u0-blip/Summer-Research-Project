import configparser
import argparse

def get_args():
    # Construct the argument parser
    ap = argparse.ArgumentParser()

    # Add the arguments to the parser
    # ap.add_argument("-d", "--distance", required=True,
    # help="distance between solid")
    # ap.add_argument("-s", "--size", required=True,
    # help="size of solid")
    # ap.add_argument("-f", "--file", required=True,
    # help="file to output")
    # ap.add_argument("-v", "--visual", required=True,
    # help="whether to visualize")
    
    ap.add_argument("-s", "--section", required=True,
    help="config section")

    args = vars(ap.parse_args())
    return args

args = get_args()
config = configparser.ConfigParser()
config.read('sim.ini')
