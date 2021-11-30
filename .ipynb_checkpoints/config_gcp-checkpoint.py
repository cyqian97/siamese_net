from os.path import join

training_dir = join("..","..","datasets","tutorial", "train")
testing_dir = join("..","..","datasets", "tutorial","test")
training_csv = "train_data.csv"
testing_csv = "test_data.csv"
batch_size = 128
epochs = 20
learning_rate = 1e-4
weight_decay = 0.0005
img_height = 155
img_width = 220
