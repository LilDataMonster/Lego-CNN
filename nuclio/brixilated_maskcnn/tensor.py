import os
import sys
import os.path
import re
import requests
import shutil
import tarfile
import traceback
import json
import threading

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
import tensorflow as tf

sys.path.append('/lego_cnn')

#from lego_cnn.mrcnn import model as modellib
import mrcnn.model as modellib
from samples.brixilated_lego import config
#from samples.brixilated_lego import lego


def classify(context, event):

    # we're going to need a unique temporary location to handle each event,
    # as we download a file as part of each function invocation
    temp_dir = Helpers.create_temporary_dir(context, event)

    # wrap everything with error handling such that any exception raised
    # at any point will still return a proper response
    try:

        context.logger.info(f'Done Loading: {FunctionState.done_loading}')
        # if we're not ready to handle this request yet, deny it
        if not FunctionState.done_loading:
            context.logger.warn_with('Model data not done loading yet, denying request')
            raise NuclioResponseError('Model data not loaded yet, cannot serve this request',
                                      requests.codes.service_unavailable)

        # read the event's body to determine the target image URL
        # TODO: in the future this can also take binary image data if provided with an appropriate content-type
        image_url = event.body.decode('utf-8').strip()

        # download the image to our temporary location
        image_target_path = os.path.join(temp_dir, 'downloaded_image.jpg')
        Helpers.download_file(context, image_url, image_target_path)

        # run the inference on the image
        results = Helpers.run_inference(context, image_target_path, 5, 0.3)

        response = {}
        response['class_ids'] = results['class_ids'].tolist()
        response['rois'] = results['rois'].tolist()
        response['scores'] = results['scores'].tolist()
        context.logger.info(f'response: {response}')

        # return a response with the result
        return context.Response(body=json.dumps(response),
                                headers={},
                                content_type='application/json',
                                status_code=requests.codes.ok)

    # convert any NuclioResponseError to a response to be returned from our handler.
    # the response's description and status will appropriately convey the underlying error's nature
    except NuclioResponseError as error:
        return error.as_response(context)

    # if anything we didn't count on happens, respond with internal server error
    except Exception as error:
        context.logger.warn_with('Unexpected error occurred, responding with internal server error',
                                 exc=str(error))

        message = 'Unexpected error occurred: {0}\n{1}'.format(error, traceback.format_exc())
        return NuclioResponseError(message).as_response(context)

    # clean up after ourselves regardless of whether we succeeded or failed
    finally:
        shutil.rmtree(temp_dir)


class NuclioResponseError(Exception):

    def __init__(self, description, status_code=requests.codes.internal_server_error):
        self._description = description
        self._status_code = status_code

    def as_response(self, context):
        return context.Response(body=self._description,
                                headers={},
                                content_type='text/plain',
                                status_code=self._status_code)


class InferenceConfig(config.LegoConfig().__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


class FunctionState(object):
    """
    This class has classvars that are set by methods invoked during file import,
    such that handler invocations can re-use them.
    """

    # holds the Mask R-CNN model
    model = None

    is_loading = False

#    # holds the TensorFlow graph def
#    graph = None

    # holds the node id to human string mapping
    node_lookup = None

    # holds a boolean indicating if we're ready to handle an invocation or haven't finished yet
    done_loading = False


class Paths(object):

    # Mask R-CNN paths
    model_path = os.getenv('MODEL_PATH', os.path.join(os.sep, 'logs'))
    weights_path = os.getenv('WEIGHTS_PATH', os.path.join(os.sep))
    weights_filename = os.getenv('WEIGHTS_FILENAME', 'mask_rcnn_lego.h5')


class Helpers(object):

    @staticmethod
    def create_temporary_dir(context, event):
        """
        Creates a uniquely-named temporary directory (based on the given event's id) and returns its path.
        """
        temp_dir = '/tmp/nuclio-event-{0}'.format(event.id)
        os.makedirs(temp_dir)

        context.logger.debug_with('Created temporary directory', path=temp_dir)

        return temp_dir

    @staticmethod
    def run_inference(context, image_path, num_predictions, confidence_threshold):
        """
        Runs inference on the image in the given path.
        Returns a list of up to N=num_prediction tuples (prediction human name, confidence score).
        Only takes predictions whose confidence score meets the provided confidence threshold.
        """

        # read image
        image = tf.keras.preprocessing.image.load_img(image_path)
        image_data = tf.keras.preprocessing.image.img_to_array(image)

        # predict
        model = FunctionState.model
        results = model.detect([image_data], verbose=0)[0]

        return results

    @staticmethod
    def on_import():
        """
        This function is called when the file is imported, so that model data
        is loaded to memory only once per function deployment.
        """

        # load the Mask R-CNN Model
        FunctionState.model = Helpers.load_maskrcnn()

        # signal that we're ready
        FunctionState.done_loading = True

    @staticmethod
    def load_maskrcnn():
        """
        Initializes the Mask R-CNN model
        """

        weights_path = os.path.join(Paths.weights_path, Paths.weights_filename)
        if not os.path.isfile(weights_path):
            raise NuclioResponseError('Failed to load weights', requests.codes.service_unavailable)

        config = InferenceConfig()
        model = modellib.MaskRCNN(mode="inference", model_dir="", config=config)
        model.load_weights(weights_path, by_name=True)

        return model

    @staticmethod
    def download_file(context, url, target_path):
        """
        Downloads the given remote URL to the specified path.
        """
        # make sure the target directory exists
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        try:
            with requests.get(url, stream=True) as response:
                response.raise_for_status()
                with open(target_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
        except Exception as error:
            if context is not None:
                context.logger.warn_with('Failed to download file',
                                         url=url,
                                         target_path=target_path,
                                         exc=str(error))
            raise NuclioResponseError('Failed to download file: {0}'.format(url),
                                      requests.codes.service_unavailable)
        if context is not None:
            context.logger.info_with('Downloaded file successfully',
                                     size_bytes=os.stat(target_path).st_size,
                                     target_path=target_path)


# perform the loading in another thread to not block import - the function
# handler will gracefully decline requests until we're ready to handle them
t = threading.Thread(target=Helpers.on_import)
t.start()
