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
#from samples.brixilated_lego import config
from samples.brixilated_lego import lego


def classify(context, event):

    #test = {
    #        '1': 2
    #        }
    #return context.Response(body=str(os.listdir('/lego_cnn/samples/brixilated_lego')),
    #                        headers={},
    #                        content_type='text/plain',
    #                        status_code=requests.codes.ok)
    context.logger.info(f'lego_cnn ls: {os.listdir("/lego_cnn/samples/brixilated_lego")}')

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
            #if not FunctionState.is_loading:
            #    #Helpers.on_import()
            #    t = threading.Thread(target=Helpers.on_import)
            #    t.start()
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

        # return a response with the result
        return context.Response(body=json.dumps(results),
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


class InferenceConfig(lego.LegoConfig().__class__):
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

    ## the directory in the deployed function container where the data model is saved
    #model_dir = os.getenv('MODEL_DIR', '/tmp/tfmodel/')

    ## paths of files within the model archive used to create the graph
    #label_lookup_path = os.path.join(model_dir,
    #                                 os.getenv('LABEL_LOOKUP_FILENAME',
    #                                           'imagenet_synset_to_human_label_map.txt'))

    #uid_lookup_path = os.path.join(model_dir,
    #                               os.getenv('UID_LOOKUP_FILENAME',
    #                                         'imagenet_2012_challenge_label_map_proto.pbtxt'))

    #graph_def_path = os.path.join(model_dir,
    #                              os.getenv('GRAPH_DEF_FILENAME',
    #                                        'classify_image_graph_def.pb'))

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

#        # read the image binary data
#        with tf.gfile.FastGFile(image_path, 'rb') as f:
#            image_data = f.read()

        # read image
        image = tf.keras.preprocessing.image.load_img(image_path)
        image_data = tf.keras.preprocessing.image.img_to_array(image)
        print(np.shape(image_data))

        # predict
        model = FunctionState.model
        results = model.detect([image_data], verbose=0)[0]

        return results

#        # run the graph's softmax tensor on the image data
#        with tf.Session(graph=FunctionState.graph) as session:
#            softmax_tensor = session.graph.get_tensor_by_name('softmax:0')
#            predictions = session.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})
#            predictions = np.squeeze(predictions)
#
#        results = []
#
#        # take the num_predictions highest scoring predictions
#        top_predictions = reversed(predictions.argsort()[-num_predictions:])
#
#        # look up each predicition's human-readable name and add it to the
#        # results if it meets the confidence threshold
#        for node_id in top_predictions:
#            name = FunctionState.node_lookup[node_id]
#
#            score = predictions[node_id]
#            meets_threshold = score > confidence_threshold
#
#            # tensorflow's float32 must be converted to float before logging, not JSON-serializable
#            context.logger.info_with('Found prediction',
#                                     name=name,
#                                     score=float(score),
#                                     meets_threshold=meets_threshold)
#
#            if meets_threshold:
#                results.append((name, score))
#
#        return results

    @staticmethod
    def on_import():
        """
        This function is called when the file is imported, so that model data
        is loaded to memory only once per function deployment.
        """

        print('on_import()!')

        FunctionState.is_loading = True

        # load the Mask R-CNN Model
        FunctionState.model = Helpers.load_maskrcnn()

        ## load the graph def from trained model data
        #FunctionState.graph = Helpers.load_graph_def()

        ## load the node ID to human-readable string mapping
        #FunctionState.node_lookup = Helpers.load_node_lookup()

        # signal that we're ready
        FunctionState.done_loading = True
        FunctionState.is_loading = False

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
        model.keras_model._make_predict_function()

        return model

#    @staticmethod
#    def load_graph_def():
#        """
#        Imports the GraphDef data into TensorFlow's default graph, and returns it.
#        """
#
#        # verify that the declared graph def file actually exists
#        if not tf.gfile.Exists(Paths.graph_def_path):
#            raise NuclioResponseError('Failed to find graph def file', requests.codes.service_unavailable)
#
#        # load the TensorFlow GraphDef
#        with tf.gfile.FastGFile(Paths.graph_def_path, 'rb') as f:
#            graph_def = tf.GraphDef()
#            graph_def.ParseFromString(f.read())
#
#            tf.import_graph_def(graph_def, name='')
#
#        return tf.get_default_graph()

#    @staticmethod
#    def load_node_lookup():
#        """
#        Composes the mapping between node IDs and human-readable strings. Returns the composed mapping.
#        """
#
#        # load the mappings from which we can build our mapping
#        string_uid_to_labels = Helpers._load_label_lookup()
#        node_id_to_string_uids = Helpers._load_uid_lookup()
#
#        # compose the final mapping of integer node ID to human-readable string
#        result = {}
#        for node_id, string_uid in node_id_to_string_uids.items():
#            label = string_uid_to_labels.get(string_uid)
#
#            if label is None:
#                raise NuclioResponseError('Failed to compose node lookup')
#
#            result[node_id] = label
#
#        return result

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

#    @staticmethod
#    def _load_label_lookup():
#        """
#        Loads and parses the mapping between string UIDs and human-readable strings. Returns the parsed mapping.
#        """
#
#        # verify that the declared label lookup file actually exists
#        if not tf.gfile.Exists(Paths.label_lookup_path):
#            raise NuclioResponseError('Failed to find Label lookup file', requests.codes.service_unavailable)
#
#        # load the raw mapping data
#        with tf.gfile.GFile(Paths.label_lookup_path) as f:
#            lookup_lines = f.readlines()
#
#        result = {}
#
#        # parse the raw data to a mapping between string UIDs and labels
#        # each line is expected to look like this:
#        # n12557064     kidney bean, frijol, frijole
#        line_pattern = re.compile(r'(n\d+)\s+([ \S,]+)')
#
#        for line in lookup_lines:
#            matches = line_pattern.findall(line)
#
#            # extract the uid and label from the matches
#            # in our example, uid will be "n12557064" and label will be "kidney bean, frijol, frijole"
#            uid = matches[0][0]
#            label = matches[0][1]
#
#            # insert the UID and label to our mapping
#            result[uid] = label
#
#        return result

    @staticmethod
    def _load_uid_lookup():
        """
        Loads and parses the mapping between node IDs and string UIDs. Returns the parsed mapping.
        """

        # verify that the declared uid lookup file actually exists
        if not tf.gfile.Exists(Paths.uid_lookup_path):
            raise NuclioResponseError('Failed to find UID lookup file', requests.codes.service_unavailable)

        # load the raw mapping data
        with tf.gfile.GFile(Paths.uid_lookup_path) as f:
            lookup_lines = f.readlines()

        result = {}

        # parse the raw data to a mapping between integer node IDs and string UIDs
        # this file is expected to contains entries such as this:
        #
        # entry
        # {
        #   target_class: 443
        #   target_class_string: "n01491361"
        # }
        #
        # to parse it, we'll iterate over the lines, and for each line that begins with "  target_class:"
        # we'll assume that the next line has the corresponding "target_class_string"
        for i, line in enumerate(lookup_lines):

            # we found a line that starts a new entry in our mapping
            if line.startswith('  target_class:'):
                # target_class represents an integer value for node ID - convert it to an integer
                target_class = int(line.split(': ')[1])

                # take the string UID from the next line,
                # and clean up the quotes wrapping it (and the trailing newline)
                next_line = lookup_lines[i + 1]
                target_class_string = next_line.split(': ')[1].strip('"\n ')

                # insert the node ID and string UID to our mapping
                result[target_class] = target_class_string

        return result


print('starting thread')
# perform the loading in another thread to not block import - the function
# handler will gracefully decline requests until we're ready to handle them
t = threading.Thread(target=Helpers.on_import)
t.start()
