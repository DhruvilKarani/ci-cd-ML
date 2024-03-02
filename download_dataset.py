import tensorflow as tf

# Download the MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Convert the data to TF dataset format
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

# Save the datasets to a directory
data_dir = '/data'
tf.data.experimental.save(train_dataset, data_dir + '/train_dataset')
tf.data.experimental.save(test_dataset, data_dir + '/test_dataset')
