import numpy as np
import bentoml
from bentoml.io import NumpyNdarray, JSON

#BENTO_MODEL_TAG = "onnx_model:z3jrnixeno2mxlg6"
BENTO_MODEL_TAG = "onnx_model:latest"
runner = bentoml.onnx.get(BENTO_MODEL_TAG).to_runner()

loan_service = bentoml.Service("loan_evaluator", runners=[runner])

@loan_service.api(input=NumpyNdarray(), output=JSON())
async def classify(input_array: np.ndarray)->np.ndarray:
    return await runner.run.async_run(input_array)