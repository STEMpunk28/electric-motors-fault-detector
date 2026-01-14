# Electric motors fault detector

Exploration of different deep learning models for detecting abnormal sound signals from electric motors.

## Dataset

[MIMII Dataset](https://zenodo.org/records/3384388) is used to train, evaluate and test this project.

## Preprocessing
[Segment.py](Preprocess\Segment.py) contains the code to change audio files to image files, used through [Mass_Transform.py](Preprocess\Mass_Transform.py) to apply to each file in a more quick manner. Finally, [split.py](Preprocess\split.py) contains the code used to segment the dataset into the three subdataset used.

## Training
+ [Dataset_loader.py](Training\Dataset_loader.py) makes sure to load the dataset in the same manner for all models, except for MobileNetV2_transfer.py and Transformer_training.py

+ [MobileNetV2_transfer.py](Training/MobileNetV2_transfer.py) is an implementation of [TensorFlow fine-tuning tutorial](https://www.tensorflow.org/tutorials/images/transfer_learning) to use as an accuracy base for comparisions.

+ [CCN_training.py](Training\CNN_training.py), [LSTM_training.py](Training\LSTM_training.py), [BiLSTM_training.py](Training\BiLSTM_training.py), [CNN_LSTM_training.py](Training\CNN_LSTM_training.py) and [CNN_BiLSTM_training.py](Training/CNN_BiLSTM_training.py) models all work the same way, outputting the best weights into a .keras file and a .csv with the evolution of accuraccy and loss.

+ [Transformer_training.py](Training\Transformer_training.py) is an exploration using ViT transformers using PyTorch, not included on the main exploration for a lack of hardware and time.

## Graphs

[Architecture.py](Graphs\Architecture.py) generates a visualization of the model layers, [Epoch_graphs.py](Graphs\Epoch_graphs.py) shows the evolution of accuracy and loss that was logged in the .csv files, [Matrix_graphs.py](Graphs\Matrix_graphs.py) create confusion matrix for every .keras model and [Test_graphs.py](Graphs\Test_graphs.py) create comparision graphs of all models using f1-score, accuracy and precision.