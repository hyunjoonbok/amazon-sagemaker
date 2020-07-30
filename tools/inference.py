import io
import json
import base64
import numpy as np
import tensorflow as tf
from collections import namedtuple
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from PIL import Image

# restricting memory growth
physical_gpus = tf.config.experimental.list_physical_devices('GPU')
if physical_gpus:
    try:
    # Currently, memory growth needs to be the same across GPUs
        for gpu in physical_gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(physical_gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
        print(e)

HEIGHT = 224
WIDTH  = 224

Context = namedtuple('Context',
                     'model_name, model_version, method, rest_uri, grpc_uri, '
                     'custom_attributes, request_content_type, accept_header')

def input_handler(data, context):
    """ Pre-process request input before it is sent to TensorFlow Serving REST API
    Args:
        data (obj): the request data, in format of dict or string
        context (Context): an object containing request and configuration details
    Returns:
        (dict): a JSON-serializable dict that contains request body and headers
    """
    if context.request_content_type == 'application/x-image':
        # pass through json (assumes it's correctly formed)
        #read image as bytes
        image_as_bytes = io.BytesIO(data.read())
        img = Image.open(image_as_bytes)
        img = img.resize((WIDTH, HEIGHT))
        # convert PIL image instance to numpy array
        img_array = image.img_to_array(img, data_format = "channels_first")
        # the image is now in an array of shape (3, 224, 224)
        # need to expand it to (1, 3, 224, 224) as it's expecting a list
        expanded_img_array = tf.expand_dims(img_array, axis=0)
        #preprocessing the image array with channel first
        preprocessed_img = preprocess_input(expanded_img_array, data_format = "channels_first")
        #converting to numpy list
        preprocessed_img_lst = preprocessed_img.numpy().tolist()
        return json.dumps({"instances": preprocessed_img_lst})
    else:
        _return_error(415, 'Unsupported content type "{}"'.format(context.request_content_type or 'Unknown'))


def output_handler(data, context):
    """Post-process TensorFlow Serving output before it is returned to the client.
    Args:
        data (obj): the TensorFlow serving response
        context (Context): an object containing request and configuration details
    Returns:
        (bytes, string): data to return to client, response content type
    """
    if data.status_code != 200:
        raise ValueError(data.content.decode('utf-8'))

    response_content_type = context.accept_header
    prediction = data.content
    return prediction,response_content_type

def _return_error(code, message):
    raise ValueError('Error: {}, {}'.format(str(code), message))