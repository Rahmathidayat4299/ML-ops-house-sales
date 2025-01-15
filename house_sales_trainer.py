import tensorflow as tf
import tensorflow_transform as tft
from tensorflow.keras import layers
import os
from tfx.components.trainer.fn_args_utils import FnArgs

# Define constants for label and feature keys
LABEL_KEY = "price"
FEATURE_KEYS = ["bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors", "waterfront", "view", "condition", "grade",
                "sqft_above", "sqft_basement", "yr_built", "yr_renovated", "sqft_living15", "sqft_lot15"]

# Rename transformed features to append '_xf' to the original feature names
def transformed_name(key):
    return key + "_xf"

# Function to read compressed TFRecord files
def gzip_reader_fn(filenames):
    return tf.data.TFRecordDataset(filenames, compression_type='GZIP')

# Create a batched dataset from transformed features
def input_fn(file_pattern, tf_transform_output, num_epochs, batch_size=64) -> tf.data.Dataset:
    # Load transformed feature specification from TensorFlow Transform
    transform_feature_spec = tf_transform_output.transformed_feature_spec().copy()

    # Create a batched dataset with specified features, labels, and batch size
    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transform_feature_spec,
        reader=gzip_reader_fn,
        num_epochs=num_epochs,
        label_key=transformed_name(LABEL_KEY)  # Use transformed label key
    )
    return dataset

# Build the deep learning model for regression (predicting price)
def model_builder():
    # Define input layer for numerical features (transformed feature)
    inputs = {transformed_name(feature): tf.keras.Input(shape=(1,), name=transformed_name(feature), dtype=tf.float32)
              for feature in FEATURE_KEYS}
    concatenated = layers.Concatenate()(list(inputs.values()))

    # Add fully connected (dense) layers
    x = layers.Dense(128, activation="relu")(concatenated)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dense(32, activation="relu")(x)

    # Output layer for regression (no activation function, as we are predicting a continuous value)
    outputs = layers.Dense(1)(x)

    # Compile the model with mean squared error loss (for regression)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        loss='mean_squared_error',
        optimizer=tf.keras.optimizers.Adam(0.01),
        metrics=['mean_absolute_error']
    )

    # Print model summary
    model.summary()
    return model

# Function to preprocess raw request data for deployment
def _get_serve_tf_examples_fn(model, tf_transform_output):
    # Attach TFT transform features layer to the model
    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function
    def serve_tf_examples_fn(serialized_tf_examples):
        # Parse raw features and apply transformations
        feature_spec = tf_transform_output.raw_feature_spec()
        # feature_spec.pop(LABEL_KEY)  # Remove label key from feature spec - DO NOT POP
        parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)
        transformed_features = model.tft_layer(parsed_features)
        return model(transformed_features)

    return serve_tf_examples_fn

# Function to define filepath for saving the model
def get_keras_filepath(model_dir):
    return os.path.join(model_dir, 'model.keras')

# Main function for training
def run_fn(fn_args: FnArgs) -> None:
    # Load the transform graph output for data transformation
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)

    # Create training and validation datasets
    train_set = input_fn(fn_args.train_files, tf_transform_output, num_epochs=10)
    val_set = input_fn(fn_args.eval_files, tf_transform_output, num_epochs=10)

    # Define TensorBoard callback
    log_dir = os.path.join(os.path.dirname(fn_args.serving_model_dir), 'logs')
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, update_freq='batch', profile_batch=0)

    # Early stopping callback
    es = tf.keras.callbacks.EarlyStopping(monitor='val_mean_absolute_error', mode='min', verbose=1, patience=10)

    # Model checkpoint callback
    filepath = get_keras_filepath(fn_args.serving_model_dir)
    mc = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_mean_absolute_error',
                                            mode='min', verbose=1, save_best_only=True)

    # Build the model
    model = model_builder()

    # Train the model
    model.fit(
        x=train_set,
        validation_data=val_set,
        callbacks=[tensorboard_callback, es, mc],
        steps_per_epoch=1000,
        validation_steps=1000,
        epochs=10
    )

    # Define model signatures for serving
    signatures = {
        'serving_default': _get_serve_tf_examples_fn(model, tf_transform_output).get_concrete_function(
            tf.TensorSpec(shape=[None], dtype=tf.string, name='examples')
        )
    }

    # Save the model
    tf.saved_model.save(model, fn_args.serving_model_dir, signatures=signatures)

# %%writefile house_sales_trainer.py
# import tensorflow as tf
# import tensorflow_transform as tft
# from tensorflow.keras import layers
# import os
# from tfx.components.trainer.fn_args_utils import FnArgs

# # Define constants for label and feature keys
# LABEL_KEY = "price"
# FEATURE_KEYS = ["bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors", "waterfront", "view", "condition", "grade",
#                 "sqft_above", "sqft_basement", "yr_built", "yr_renovated", "sqft_living15", "sqft_lot15"]

# # Rename transformed features to append '_xf' to the original feature names
# def transformed_name(key):
#     return key + "_xf"

# # Function to read compressed TFRecord files
# def gzip_reader_fn(filenames):
#     return tf.data.TFRecordDataset(filenames, compression_type='GZIP')

# # Create a batched dataset from transformed features
# def input_fn(file_pattern, tf_transform_output, num_epochs, batch_size=64) -> tf.data.Dataset:
#     # Load transformed feature specification from TensorFlow Transform
#     transform_feature_spec = tf_transform_output.transformed_feature_spec().copy()

#     # Create a batched dataset with specified features, labels, and batch size
#     dataset = tf.data.experimental.make_batched_features_dataset(
#         file_pattern=file_pattern,
#         batch_size=batch_size,
#         features=transform_feature_spec,
#         reader=gzip_reader_fn,
#         num_epochs=num_epochs,
#         label_key=transformed_name(LABEL_KEY)  # Use transformed label key
#     )
#     return dataset

# # Build the deep learning model for regression (predicting price)
# def model_builder():
#     # Define input layer for numerical features (transformed feature)
#     inputs = {transformed_name(feature): tf.keras.Input(shape=(1,), name=transformed_name(feature), dtype=tf.float32)
#               for feature in FEATURE_KEYS}
#     concatenated = layers.Concatenate()(list(inputs.values()))

#     # Add fully connected (dense) layers
#     x = layers.Dense(128, activation="relu")(concatenated)
#     x = layers.Dense(64, activation="relu")(x)
#     x = layers.Dense(32, activation="relu")(x)

#     # Output layer for regression (no activation function, as we are predicting a continuous value)
#     outputs = layers.Dense(1)(x)

#     # Compile the model with mean squared error loss (for regression)
#     model = tf.keras.Model(inputs=inputs, outputs=outputs)
#     model.compile(
#         loss='mean_squared_error',
#         optimizer=tf.keras.optimizers.Adam(0.01),
#         metrics=['mean_absolute_error']
#     )

#     # Print model summary
#     model.summary()
#     return model

# # Function to preprocess raw request data for deployment
# def _get_serve_tf_examples_fn(model, tf_transform_output):
#     # Attach TFT transform features layer to the model
#     model.tft_layer = tf_transform_output.transform_features_layer()

#     @tf.function
#     def serve_tf_examples_fn(serialized_tf_examples):
#         # Parse raw features and apply transformations
#         feature_spec = tf_transform_output.raw_feature_spec()
#         feature_spec.pop(LABEL_KEY)  # Remove label key from feature spec
#         parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)
#         transformed_features = model.tft_layer(parsed_features)
#         return model(transformed_features)

#     return serve_tf_examples_fn

# # Function to define filepath for saving the model
# def get_keras_filepath(model_dir):
#     return os.path.join(model_dir, 'model.keras')

# # Main function for training
# def run_fn(fn_args: FnArgs) -> None:
#     # Load the transform graph output for data transformation
#     tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)

#     # Create training and validation datasets
#     train_set = input_fn(fn_args.train_files, tf_transform_output, num_epochs=10)
#     val_set = input_fn(fn_args.eval_files, tf_transform_output, num_epochs=10)

#     # Define TensorBoard callback
#     log_dir = os.path.join(os.path.dirname(fn_args.serving_model_dir), 'logs')
#     tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, update_freq='batch', profile_batch=0)

#     # Early stopping callback
#     es = tf.keras.callbacks.EarlyStopping(monitor='val_mean_absolute_error', mode='min', verbose=1, patience=10)

#     # Model checkpoint callback
#     filepath = get_keras_filepath(fn_args.serving_model_dir)
#     mc = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_mean_absolute_error',
#                                             mode='min', verbose=1, save_best_only=True)

#     # Build the model
#     model = model_builder()

#     # Train the model
#     model.fit(
#         x=train_set,
#         validation_data=val_set,
#         callbacks=[tensorboard_callback, es, mc],
#         steps_per_epoch=1000,
#         validation_steps=1000,
#         epochs=10
#     )

#     # Define model signatures for serving
#     signatures = {
#         'serving_default': _get_serve_tf_examples_fn(model, tf_transform_output).get_concrete_function(
#             tf.TensorSpec(shape=[None], dtype=tf.string, name='examples')
#         )
#     }

#     # Save the model
#     tf.saved_model.save(model, fn_args.serving_model_dir, signatures=signatures)
