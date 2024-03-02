import tensorflow as tf
import os

# Set the path to the dataset directory
dataset_dir = '/data'

# Load the MNIST dataset
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(dataset_dir, 'train_dataset'),
    labels='inferred',
    label_mode='int',
    color_mode='grayscale',
    batch_size=32,
    image_size=(28, 28),
    validation_split=0.2,
    subset='training',
    seed=123
)

test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(dataset_dir, 'test_dataset'),
    labels='inferred',
    label_mode='int',
    color_mode='grayscale',
    batch_size=32,
    image_size=(28, 28),
    validation_split=0.2,
    subset='validation',
    seed=123
)

# Define and compile the model
model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.Dense(10)
])

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# Train the model
model.fit(train_dataset, validation_data=test_dataset, epochs=10)

# Evaluate the model and print metrics
loss, accuracy = model.evaluate(test_dataset)
print("Loss:", loss)
print("Accuracy:", accuracy)


# Save the model
model.save('/model')

