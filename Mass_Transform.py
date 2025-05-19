import sys
import os

if __name__ == '__main__':
    input_folder = sys.argv[1]
    output_folder = sys.argv[2]
    # Call Segment using the given folder
    i = 0
    for file in os.listdir(input_folder):
        os.system(f"python Segment.py {file} {output_folder} {i}")
        i+=1