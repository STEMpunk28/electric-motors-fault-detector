import os

# Training set path
PATH=""
# Model weights saving path
WEIGHTS_PATH=""
# Model architecture graph saving path
ARCH_PATH=""
# Training logs saving path
LOG_PATH=""
# Graphs saving path
GRAPH_PATH=""
# Normal class label
NORMAL_CLASS=''

# Create directories if they don't exist
os.makedirs(WEIGHTS_PATH, exist_ok=True)
os.makedirs(GRAPH_PATH, exist_ok=True)
os.makedirs(ARCH_PATH, exist_ok=True)
os.makedirs(LOG_PATH, exist_ok=True)