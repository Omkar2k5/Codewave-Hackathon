import tensorflow as tf
import os
import json
import pickle
from datetime import datetime
from absl import app, flags, logging
from absl.flags import FLAGS
from core.yolov4 import YOLO, decode, filter_boxes
import core.utils as utils
from core.config import cfg

flags.DEFINE_string('weights', './data/yolov4.weights', 'path to weights file')
flags.DEFINE_string('output', './checkpoints/yolov4-416', 'path to output')
flags.DEFINE_boolean('tiny', False, 'is yolo-tiny or not')
flags.DEFINE_integer('input_size', 416, 'define input size of export model')
flags.DEFINE_float('score_thres', 0.2, 'define score threshold')
flags.DEFINE_string('framework', 'tf', 'define what framework do you want to convert (tf, trt, tflite)')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_boolean('save_metadata', True, 'save model metadata and configuration')
flags.DEFINE_boolean('export_all_formats', False, 'export model in all supported formats')
flags.DEFINE_string('export_dir', './exported_models', 'directory to save all exported models')

def create_model():
    """Create and load the YOLO model"""
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)

    input_layer = tf.keras.layers.Input([FLAGS.input_size, FLAGS.input_size, 3])
    feature_maps = YOLO(input_layer, NUM_CLASS, FLAGS.model, FLAGS.tiny)
    bbox_tensors = []
    prob_tensors = []

    if FLAGS.tiny:
        for i, fm in enumerate(feature_maps):
            if i == 0:
                output_tensors = decode(fm, FLAGS.input_size // 16, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE, FLAGS.framework)
            else:
                output_tensors = decode(fm, FLAGS.input_size // 32, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE, FLAGS.framework)
            bbox_tensors.append(output_tensors[0])
            prob_tensors.append(output_tensors[1])
    else:
        for i, fm in enumerate(feature_maps):
            if i == 0:
                output_tensors = decode(fm, FLAGS.input_size // 8, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE, FLAGS.framework)
            elif i == 1:
                output_tensors = decode(fm, FLAGS.input_size // 16, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE, FLAGS.framework)
            else:
                output_tensors = decode(fm, FLAGS.input_size // 32, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE, FLAGS.framework)
            bbox_tensors.append(output_tensors[0])
            prob_tensors.append(output_tensors[1])

    pred_bbox = tf.concat(bbox_tensors, axis=1)
    pred_prob = tf.concat(prob_tensors, axis=1)

    if FLAGS.framework == 'tflite':
        pred = (pred_bbox, pred_prob)
    else:
        boxes, pred_conf = filter_boxes(pred_bbox, pred_prob, score_threshold=FLAGS.score_thres,
                                      input_shape=tf.constant([FLAGS.input_size, FLAGS.input_size]))
        pred = tf.concat([boxes, pred_conf], axis=-1)

    model = tf.keras.Model(input_layer, pred)

    # Load weights if they exist
    if os.path.exists(FLAGS.weights):
        utils.load_weights(model, FLAGS.weights, FLAGS.model, FLAGS.tiny)
        logging.info(f"Loaded weights from {FLAGS.weights}")
    else:
        logging.warning(f"Weights file {FLAGS.weights} not found. Model will be saved without pre-trained weights.")

    return model, NUM_CLASS, STRIDES, ANCHORS, XYSCALE

def save_model_metadata(model, output_path, num_classes, strides, anchors, xyscale):
    """Save model metadata and configuration"""
    metadata = {
        'model_type': FLAGS.model,
        'is_tiny': FLAGS.tiny,
        'input_size': FLAGS.input_size,
        'score_threshold': FLAGS.score_thres,
        'framework': FLAGS.framework,
        'num_classes': num_classes,
        'strides': strides.tolist() if hasattr(strides, 'tolist') else strides,
        'anchors': anchors.tolist() if hasattr(anchors, 'tolist') else anchors,
        'xyscale': xyscale.tolist() if hasattr(xyscale, 'tolist') else xyscale,
        'export_timestamp': datetime.now().isoformat(),
        'tensorflow_version': tf.__version__,
        'model_summary': []
    }

    # Capture model summary
    try:
        import io
        import sys
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()
        model.summary()
        metadata['model_summary'] = buffer.getvalue().split('\n')
        sys.stdout = old_stdout
    except Exception as e:
        logging.warning(f"Could not capture model summary: {e}")

    # Save metadata as JSON
    metadata_path = f"{output_path}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    logging.info(f"Model metadata saved to {metadata_path}")
    return metadata_path

def save_tf():
    """Save model in TensorFlow SavedModel format"""
    model, num_classes, strides, anchors, xyscale = create_model()

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(FLAGS.output), exist_ok=True)

    # Display model summary
    print("\nModel Summary:")
    model.summary()

    # Save the model
    model.save(FLAGS.output)
    logging.info(f"Model saved to {FLAGS.output}")

    # Save metadata if requested
    if FLAGS.save_metadata:
        save_model_metadata(model, FLAGS.output, num_classes, strides, anchors, xyscale)

    return model

def save_tflite(model):
    """Convert and save model in TensorFlow Lite format"""
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()

        tflite_path = FLAGS.output.replace('.h5', '.tflite') + '.tflite'
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)

        logging.info(f"TensorFlow Lite model saved to {tflite_path}")
        return tflite_path
    except Exception as e:
        logging.error(f"Failed to convert to TensorFlow Lite: {e}")
        return None

def save_h5(model):
    """Save model in HDF5 format"""
    try:
        h5_path = FLAGS.output.replace('.pb', '.h5') + '.h5'
        model.save(h5_path, save_format='h5')
        logging.info(f"HDF5 model saved to {h5_path}")
        return h5_path
    except Exception as e:
        logging.error(f"Failed to save HDF5 model: {e}")
        return None

def export_all_formats():
    """Export model in all supported formats"""
    model, num_classes, strides, anchors, xyscale = create_model()

    # Create export directory
    os.makedirs(FLAGS.export_dir, exist_ok=True)

    # Base filename
    base_name = f"{FLAGS.model}{'_tiny' if FLAGS.tiny else ''}_{FLAGS.input_size}"

    exported_files = []

    # 1. SavedModel format
    savedmodel_path = os.path.join(FLAGS.export_dir, f"{base_name}_savedmodel")
    model.save(savedmodel_path)
    exported_files.append(savedmodel_path)
    logging.info(f"SavedModel exported to {savedmodel_path}")

    # 2. HDF5 format
    h5_path = os.path.join(FLAGS.export_dir, f"{base_name}.h5")
    try:
        model.save(h5_path, save_format='h5')
        exported_files.append(h5_path)
        logging.info(f"HDF5 model exported to {h5_path}")
    except Exception as e:
        logging.error(f"Failed to export HDF5: {e}")

    # 3. TensorFlow Lite format
    tflite_path = os.path.join(FLAGS.export_dir, f"{base_name}.tflite")
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()

        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        exported_files.append(tflite_path)
        logging.info(f"TensorFlow Lite model exported to {tflite_path}")
    except Exception as e:
        logging.error(f"Failed to export TensorFlow Lite: {e}")

    # 4. Save metadata
    if FLAGS.save_metadata:
        metadata_path = save_model_metadata(model, os.path.join(FLAGS.export_dir, base_name),
                                          num_classes, strides, anchors, xyscale)
        exported_files.append(metadata_path)

    # 5. Save model weights separately
    weights_path = os.path.join(FLAGS.export_dir, f"{base_name}_weights.h5")
    try:
        model.save_weights(weights_path)
        exported_files.append(weights_path)
        logging.info(f"Model weights exported to {weights_path}")
    except Exception as e:
        logging.error(f"Failed to export weights: {e}")

    # Create export summary
    summary_path = os.path.join(FLAGS.export_dir, f"{base_name}_export_summary.txt")
    with open(summary_path, 'w') as f:
        f.write(f"Model Export Summary\n")
        f.write(f"==================\n\n")
        f.write(f"Export Date: {datetime.now().isoformat()}\n")
        f.write(f"Model Type: {FLAGS.model}\n")
        f.write(f"Tiny Model: {FLAGS.tiny}\n")
        f.write(f"Input Size: {FLAGS.input_size}\n")
        f.write(f"Score Threshold: {FLAGS.score_thres}\n")
        f.write(f"Framework: {FLAGS.framework}\n\n")
        f.write(f"Exported Files:\n")
        for file_path in exported_files:
            if os.path.exists(file_path):
                size = os.path.getsize(file_path) if os.path.isfile(file_path) else "Directory"
                f.write(f"  - {file_path} ({size} bytes)\n")

    exported_files.append(summary_path)
    logging.info(f"Export summary saved to {summary_path}")

    return exported_files

def main(_argv):
    """Main function to handle model saving and export"""
    try:
        if FLAGS.export_all_formats:
            exported_files = export_all_formats()
            print(f"\nSuccessfully exported model in multiple formats:")
            for file_path in exported_files:
                print(f"  - {file_path}")
        else:
            model = save_tf()
            print(f"\nModel successfully saved to: {FLAGS.output}")

            # Additional format exports based on framework flag
            if FLAGS.framework == 'tflite':
                save_tflite(model)
            elif FLAGS.framework == 'h5':
                save_h5(model)

    except Exception as e:
        logging.error(f"Error during model export: {e}")
        raise

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
