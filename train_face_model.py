import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os
import pathlib
import matplotlib.pyplot as plt
import mlflow
import mlflow.tensorflow

def train_and_convert_model(dataset_dir='dataset', output_model='face_model_quantized.tflite', epochs=20):
    # Setup MLflow - Use environment variable if available (for CI/CD), otherwise local Docker
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("Face_Recognition_Auth")
    
    with mlflow.start_run():
        data_dir = pathlib.Path(dataset_dir)
        
        if not data_dir.exists():
            print(f"Error: Dataset directory '{dataset_dir}' not found.")
            print("Please run face_dataset_collector.py first.")
            return

        batch_size = 32
        img_height = 96
        img_width = 96

        mlflow.log_params({
            "epochs": epochs,
            "batch_size": batch_size,
            "img_height": img_height,
            "img_width": img_width,
            "color_mode": "grayscale"
        })

        print("Loading dataset...")
        train_ds = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=(img_height, img_width),
            batch_size=batch_size,
            color_mode='grayscale'
        )

        val_ds = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=0.2,
            subset="validation",
            seed=123,
            image_size=(img_height, img_width),
            batch_size=batch_size,
            color_mode='grayscale'
        )

        class_names = train_ds.class_names
        num_classes = len(class_names)
        print(f"Classes found: {class_names}")
        mlflow.log_param("num_classes", num_classes)
        
        # Save class names to a C header file for ESP32
        with open("labels.h", "w") as f:
            f.write("#ifndef LABELS_H\n#define LABELS_H\n\n")
            f.write(f"const int kNumClasses = {num_classes};\n")
            f.write("const char* kCategoryLabels[kNumClasses] = {\n")
            for name in class_names:
                f.write(f'    "{name}",\n')
            f.write("};\n\n#endif // LABELS_H")
        print("Saved labels.h")

        normalization_layer = layers.Rescaling(1./255)
        normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
        normalized_val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

        data_augmentation = tf.keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
        ])

        model = models.Sequential([
            layers.Input(shape=(img_height, img_width, 1)),
            
            data_augmentation,

            layers.Conv2D(16, 3, strides=2, padding='same', activation='relu'), 
            layers.MaxPooling2D(),
            
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            
            layers.Dropout(0.2), 
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(num_classes)
        ])

        model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])

        model.summary()

        print("Starting training...")
        mlflow.tensorflow.autolog() # Continue using autolog as primary
        
        history = model.fit(
            normalized_ds,
            validation_data=normalized_val_ds,
            epochs=epochs
        )

        # MANUAL LOG FALLBACK: Ensure metrics appear in MLflow
        print("Logging final metrics to MLflow...")
        final_acc = history.history['accuracy'][-1]
        final_loss = history.history['loss'][-1]
        final_val_acc = history.history['val_accuracy'][-1]
        final_val_loss = history.history['val_loss'][-1]
        
        mlflow.log_metrics({
            "final_accuracy": final_acc,
            "final_loss": final_loss,
            "final_val_accuracy": final_val_acc,
            "final_val_loss": final_val_loss
        })

        print("Converting to Quantized TFLite...")

        def representative_data_gen():
            for input_value, _ in normalized_ds.take(100):
                yield [input_value]

        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_data_gen
        
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

        tflite_model = converter.convert()

        with open(output_model, 'wb') as f:
            f.write(tflite_model)
        
        mlflow.log_artifact(output_model)
        mlflow.log_artifact("labels.h")
        
        print(f"Success! Model saved to {output_model}")
        print(f"Model Size: {len(tflite_model) / 1024:.2f} KB")

if __name__ == "__main__":
    train_and_convert_model()
