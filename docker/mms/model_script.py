from collections import namedtuple
import glob
import json
import logging
import os
import re

import lightgbm as lgb
import numpy as np

class ModelHandler(object):
    """
    A lightGBM Model handler implementation.
    """

    def __init__(self):
        self.initialized = False
        self.model = None

    def initialize(self, context):
        """
        Initialize model. This will be called during model loading time
        :param context: Initial context contains model server system properties.
        :return: None
        """
        self.initialized = True
        properties = context.system_properties
        model_dir = properties.get("model_dir") 
        # assumed the model artifact only contains one file
        model_file = glob.glob(f"{model_dir}/*")[0]
        self.model = lgb.Booster(model_file=os.path.join(model_dir, model_file))
       

    def preprocess(self, request):
        """
        Transform raw input into model input data.
        :param request: list of raw requests
        :return: list of preprocessed model input data
        """        
        print(f"request object: {request}")
        payload = request[0]['body']
        print(f"payload: {payload}")
        # split with default "\n" - csv format
        arr = payload.decode().split()
        # stacking the rows
        rows = list()
        for row in arr:
            rows.append(np.fromstring(row, dtype=np.float64, sep=',' ))

        data = np.vstack(rows)
        print(f"data: {data}")
        return data

    def inference(self, model_input):
        """
        Internal inference methods
        :param model_input: transformed model input data list
        :return: list of inference output in numpy array
        """
        prediction = self.model.predict(model_input)
        return prediction

    def postprocess(self, inference_output):
        """
        Post processing step - converts predictions to str
        :param inference_output: predictions as numpy
        :return: list of inference output as string
        """

        return [str(inference_output.tolist())]
        
    def handle(self, data, context):
        """
        Call preprocess, inference and post-process functions
        :param data: input data
        :param context: mms context
        """
        
        model_input = self.preprocess(data)
        model_out = self.inference(model_input)
        print(f"model output: {model_out}")
        return self.postprocess(model_out)

_service = ModelHandler()


def handle(data, context):
    if not _service.initialized:
        print("initialization...")
        _service.initialize(context)

    if data is None:
        return None

    return _service.handle(data, context)