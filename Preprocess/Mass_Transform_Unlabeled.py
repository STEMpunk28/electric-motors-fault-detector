import sys
import os

if __name__ == '__main__':
    input_folder = sys.argv[1]
    output_folder = sys.argv[2]
    i = int(sys.argv[3])

    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith(".mp3"):
                full_path = os.path.join(root, file)
                os.system('cls')
                print(f"Processing file #{i}: {full_path}")
                os.system(f"python Segment.py \"{full_path}\" \"{output_folder}\" {i}")
                i += 1