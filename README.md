# Loan Evaluator
This is a project that aims to evaluate customer's data and predict if we should approve a Loan or not.

## Usage
For fast deployment in any cloud provider, docker usage is highly recommended, you can Download a previously created Docker Image from this [Drive Link](https://drive.google.com/file/d/1CM65O71yEWm_zCyjr7lToul4LCh4mN2j/view?usp=drive_link), and load it with `docker load -i loan_service.tar`.

If you want to dockerize the service the steps are:

0. Install the required libs with `pip intall -r requirements.txt`, and run the training pipeline with `python3 train/training.py`, this is to train the model and save it to the Bento Local Repository.
1. Enter to the `prod/` folder with `cd prod/api_service/` and run:
    ```bash
    bentoml build
    ```
    This will create a Bento, this packs an API Service with a Bento exported Model.
  
    To check the Bento Tag run:
    ```bash
    bentoml list
    ```
  
    You can test the Bento with:
    ```bash
    bentoml serve loan_evaluator:latest
    ```

2. Create the docker with:
    ```bash
    bentoml containerize loan_evaluator:<tag-id>
    ```
  
3. Run the Docker locally with:
    ```bash
    docker run --rm -p 3000:3000 loan_evaluator:<docker-tag>
    ```
    Example:
    ```bash
    docker run --rm -p 3000:3000 loan_evaluator:ap5romxeskzchlg6
    ```

4. Post to the active docker with the script using `python3 prod/example_request.py`, or you can post via postman to the url `http://localhost:3000/classify`:
    ```json
    [[1,2,3,4,1,2,3,4]]
    ```
    Response:
    ```json
      [
        [
            0
        ],
        [
            [
                0.5001351987524033,
                0.4998648012475967
            ]
        ]
    ]
    ```
5. If everything works fine, you can export the model with:
    ```bash
    docker save loan_evaluator > loan_service.tar
    ```

### Response Description
The response is a Json with a list of two lists, the first one is the model prediction, the second list are the class probabilities, the first value is the score for class 0, and the second score is for class 1. Most of the time we will use just the prediction (the first list in the response).
```python
      [
        [
            0  # Prediction  
        ],
        [
            [  # Prediction Scores
                0.5001351987524033,
                0.4998648012475967
            ]
        ]
    ]
```

## Training
The model is a `LogisticRegression` model that is trained and exported to ONNX -> Bentoml using the `TrainingPipeline` class inside the `train/training.py` file.

## Validation
The model validation is performed in the Training Pipeline, a model file, the classification report and the validation data are saved on the `output/` folder for further evaluation.

## Next Steps:
1. Create tests to evaluate codequality and prevent production bugs.
2. Perform Performance tests to check memory consumption, speed, and scalability depending on the RPMs.
3. Convert the Methods in the `training.py` file for console inputs of data paths.
