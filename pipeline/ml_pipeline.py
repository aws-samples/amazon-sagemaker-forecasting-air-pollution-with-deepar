from ml_pipeline_dependencies import *

PREPROCESSING_SCRIPT_LOCATION = "./pipeline/ml_pipeline_preprocessing.py"

def upload_preprocess_code(bucket_name):
    input_code_uri = sagemaker_session.upload_data(
        PREPROCESSING_SCRIPT_LOCATION,
        bucket = bucket_name,
        key_prefix = "preprocessing/code",
    )
    return input_code_uri
    
def create_preprocessing_step(
    execution_input,
    processing_repository_uri, 
    bucket_name, 
    input_code_uri,
    region,
    split_days = "30"
):
    preprocessing_processor = ScriptProcessor(
        command = ['python3'],
        image_uri = processing_repository_uri,
        role = role,
        instance_count = 1,
        instance_type = 'ml.m5.xlarge',
        max_runtime_in_seconds = 1200
    )

    inputs = [
        ProcessingInput(
            source = input_code_uri,
            destination = "/opt/ml/processing/input/code",
            input_name = "code"
        )
    ]

    output_data = f"s3://{bucket_name}/preprocessing/output"
    outputs = [
        ProcessingOutput(
            source = "/opt/ml/processing/output/all",
            destination = f"{output_data}/all",
            output_name = "all_data"
        ),
        ProcessingOutput(
            source = "/opt/ml/processing/output/train",
            destination = f"{output_data}/train",
            output_name = "train_data"
        ),
        ProcessingOutput(
            source = "/opt/ml/processing/output/test",
            destination = f"{output_data}/test",
            output_name = "test_data"
        )
    ]  
    
    processing_step = ProcessingStep(
        "AirQualityForecasting Preprocessing Step",
        processor = preprocessing_processor,
        job_name = execution_input["PreprocessingJobName"],
        inputs = inputs,
        outputs = outputs,
        container_arguments = ["--split-days", split_days, "--region", region, "--bucket-name", bucket_name],
        container_entrypoint = ["python3", "/opt/ml/processing/input/code/ml_pipeline_preprocessing.py"]
    )
    
    return processing_step
    
def create_hpo_step(execution_input, image_uri, bucket_name, ml_instance_type = 'ml.g4dn.8xlarge'):
    tuning_output_path = f's3://{bucket_name}/sagemaker/tuning/output'

    tuning_estimator = sagemaker.estimator.Estimator(
            sagemaker_session = sagemaker_session,
            image_uri = image_uri,
            role = role,
            instance_count = 1,
            instance_type = ml_instance_type,
            base_job_name = 'deepar-openaq-demo',
            output_path = tuning_output_path
    )
    
    hpo = dict(
        time_freq= '1H'
        ,early_stopping_patience= 40
        ,prediction_length= 48
        ,num_eval_samples= 10

        # default quantiles [0.1, 0.2, 0.3, ..., 0.9] is used
        #,test_quantiles= quantiles

        # not setting these since HPO will use range of values
        #,epochs= 400
        #,context_length= 3
        #,num_cells= 157
        #,num_layers= 4
        #,dropout_rate= 0.04
        #,embedding_dimension= 12
        #,mini_batch_size= 633
        #,learning_rate= 0.0005
    )    
    
    hpo_ranges = dict(
        epochs= IntegerParameter(1, 1000)
        ,context_length= IntegerParameter(7, 48)
        ,num_cells= IntegerParameter(30,200)
        ,num_layers= IntegerParameter(1,8)
        ,dropout_rate= ContinuousParameter(0.0, 0.2)
        ,embedding_dimension= IntegerParameter(1, 50)
        ,mini_batch_size= IntegerParameter(32, 1028)
        ,learning_rate= ContinuousParameter(.00001, .1)
    )    

    tuning_estimator.set_hyperparameters(**hpo)

    hpo_tuner = HyperparameterTuner(
        estimator = tuning_estimator, 
        objective_metric_name = 'train:final_loss',
        objective_type = 'Minimize',
        hyperparameter_ranges = hpo_ranges,
        max_jobs = 2,
        max_parallel_jobs = 1
    )

    output_data = f"s3://{bucket_name}/preprocessing/output"
    hpo_data = dict(
        train = f"{output_data}/train",
        test = f"{output_data}/test"
    )
    # as long as HPO is selected, wait for completion.
    tuning_step = TuningStep(
        "HPO Step",
        tuner = hpo_tuner,
        job_name = execution_input["TuningJobName"],
        data = hpo_data,
        wait_for_completion = True
    )    
    return tuning_step

def create_training_step(execution_input, image_uri, bucket_name, ml_instance_type = 'ml.g4dn.8xlarge'):
    training_output_path = f's3://{bucket_name}/sagemaker/training/output'
    training_estimator = sagemaker.estimator.Estimator(
            sagemaker_session = sagemaker_session,
            image_uri = image_uri,
            role = role,
            instance_count = 1,
            instance_type = ml_instance_type,
            base_job_name = 'deepar-openaq-demo',
            output_path = training_output_path
    )
    
    # best hyper parameters for tuning
    hpo = dict(
        time_freq= '1H'
        ,early_stopping_patience= 40
        ,prediction_length= 48
        ,num_eval_samples= 10
        #,test_quantiles= quantiles
        ,epochs= 400
        ,context_length= 3
        ,num_cells= 157
        ,num_layers= 4
        ,dropout_rate= 0.04
        ,embedding_dimension= 12
        ,mini_batch_size= 633
        ,learning_rate= 0.0005
    )
    training_estimator.set_hyperparameters(**hpo)    
    
    # use all the features for training.
    output_data = f"s3://{bucket_name}/preprocessing/output"
    data = dict(train = f"{output_data}/all/all_features.json")
    training_step = TrainingStep(
        "Training Step",
        estimator = training_estimator,
        data = data,
        job_name = execution_input["TrainingJobName"],
        wait_for_completion = True
    )
    return training_step

def create_model_step(execution_input, training_step):
    model_step = ModelStep(
        "Save Model",
        model = training_step.get_expected_model(),
        model_name = execution_input["ModelName"],
        result_path = "$.ModelStepResults"
    )
    return model_step

def create_existing_model_step(execution_input, image_uri):
    # for deploying existing model
    existing_model_name = f"aqf-model-{uuid.uuid1().hex}"
    existing_model = Model(
        model_data = EXISTING_MODEL_URI,
        image_uri = image_uri,
        role = role,
        name = existing_model_name
    )
    existing_model_step = ModelStep(
        "Existing Model",
        model = existing_model,
        model_name = execution_input["ModelName"]
    )
    return existing_model_step

def create_endpoint_configurgation_step(execution_input, deployment_instance_type = 'ml.c5.xlarge'):
    endpoint_config_step = EndpointConfigStep(
        "Create Model Endpoint Config",
        endpoint_config_name = execution_input["ModelName"],
        model_name = execution_input["ModelName"],
        initial_instance_count = 1,
        instance_type = deployment_instance_type
    )
    return endpoint_config_step

def create_endpoint_step(execution_input):
    endpoint_step = EndpointStep(
        "Deploy Endpoint",
        endpoint_name = execution_input["EndpointName"],
        endpoint_config_name = execution_input["ModelName"],
        update = False
    )
    return endpoint_step

def create_batch_transform_step(
    execution_input,
    bucket_name
):
    # assume we only check '0.5' quatiles predictions.
    environment_param = {
        'num_samples': 20,
        'output_types': ['quantiles'],
        'quantiles': ['0.5']
    }
    
    transformer = Transformer(
        execution_input["ModelName"],
        1,
        'ml.c5.2xlarge',
        output_path=f's3://{bucket_name}/sagemaker/batch_transform/output',
        sagemaker_session = sagemaker_session,
        strategy='MultiRecord',
        assemble_with='Line',
        env = {
            'DEEPAR_INFERENCE_CONFIG': json.dumps(environment_param)
        }
    )

    output_data = f"s3://{bucket_name}/preprocessing/output"
    transformStep = TransformStep(
        state_id = "Batch Transform Step",
        transformer = transformer,
        job_name = execution_input["TransformJobName"],
        model_name = execution_input["ModelName"],
        data = f"{output_data}/test/batch_transform_test.json",
        split_type = 'Line'
    )    
    return transformStep

def create_workflow(region, bucket_name, workflow_name, workflow_execution_role, processing_repository_uri):
    
    # Workflow Execution parameters
    execution_input = ExecutionInput(
        schema = {
            "PreprocessingJobName": str,
            "ToDoHPO": bool,
            "ToDoTraining": bool,
            "TrainingJobName": str,
            "TuningJobName": str,
            "ModelName": str,
            "EndpointName": str,
            "TransformJobName": str
        }
    )
    image_uri = sagemaker.image_uris.retrieve('forecasting-deepar', region, '1')


    # create the steps
    tuning_step = create_hpo_step(execution_input, image_uri, bucket_name)
    training_step = create_training_step(execution_input, image_uri, bucket_name)
    model_step = create_model_step(execution_input, training_step)
    existing_model_step = create_existing_model_step(execution_input, image_uri)
    endpoint_config_step = create_endpoint_configurgation_step(execution_input)
    endpoint_step = create_endpoint_step(execution_input)
    transformStep = create_batch_transform_step(execution_input, bucket_name)
    
    # Use Batch Transform instead of endpoint hosting to do one-off model prediction.
#     training_path = Chain([training_step, model_step, endpoint_config_step, endpoint_step])
#     deploy_existing_model_path = Chain([existing_model_step, endpoint_config_step, endpoint_step])
    training_path = Chain([training_step, model_step, transformStep])
    deploy_existing_model_path = Chain([existing_model_step, transformStep])

    hpo_choice = Choice(
        "To do HPO?"
    )
    training_choice = Choice(
        "To do Model Training?"
    )
    # refer to execution input variable with required format - not user friendly.
    hpo_choice.add_choice(
        rule = ChoiceRule.BooleanEquals(variable = "$$.Execution.Input['ToDoHPO']", value = True),
        next_step = tuning_step
    )
    hpo_choice.add_choice(
        rule = ChoiceRule.BooleanEquals(variable = "$$.Execution.Input['ToDoHPO']", value = False),
        next_step = training_choice
    )
    training_choice.add_choice(
        rule = ChoiceRule.BooleanEquals(variable = "$$.Execution.Input['ToDoTraining']", value = True),
        next_step = training_path
    )
    training_choice.add_choice(
        rule = ChoiceRule.BooleanEquals(variable = "$$.Execution.Input['ToDoTraining']", value = False),
        next_step = deploy_existing_model_path
    )

    input_code_uri = upload_preprocess_code(bucket_name)
    processing_step = create_preprocessing_step(execution_input, processing_repository_uri, bucket_name, input_code_uri, region)

    # catch execution exception
    failed_state_sagemaker_pipeline_step_failure = Fail(
        "ML Workflow Failed", cause = "SageMakerPipelineStepFailed"
    )
    catch_state_processing = Catch(
        error_equals = ["States.TaskFailed"],
        next_step = failed_state_sagemaker_pipeline_step_failure   
    )
    processing_step.add_catch(catch_state_processing)
    tuning_step.add_catch(catch_state_processing)
    training_step.add_catch(catch_state_processing)
    model_step.add_catch(catch_state_processing)
    endpoint_config_step.add_catch(catch_state_processing)
    endpoint_step.add_catch(catch_state_processing)
    existing_model_step.add_catch(catch_state_processing)
    
    workflow_graph = Chain([processing_step, hpo_choice])
    workflow = Workflow(
        name = workflow_name,
        definition = workflow_graph,
        role = workflow_execution_role
    )
    workflow.create()
    return workflow
    
    
def main(
    workflow_name,
    workflow_execution_role,
    processing_repository_uri,
    require_hpo,
    require_model_training
):
    # bucket_name is created in ml_pipeline_dependencies.py, which is imported at the beginning.
    workflow = create_workflow(region, bucket_name, workflow_name, workflow_execution_role, processing_repository_uri)

    # execute workflow
    preprocessing_job_name = f"aqf-preprocessing-{uuid.uuid1().hex}"
    tuning_job_name = f"aqf-tuning-{uuid.uuid1().hex}"
    training_job_name = f"aqf-training-{uuid.uuid1().hex}"
    model_job_name = f"aqf-model-{uuid.uuid1().hex}"
    endpoint_job_name = f"aqf-endpoint-{uuid.uuid1().hex}"
    batch_transform_job_name = f"aqf-transform-{uuid.uuid1().hex}"
    todoHPO = require_hpo.lower() in ['true', '1', 'yes', 't']
    todoTraining = require_model_training.lower() in ['true', '1', 'yes', 't']
    
    execution = workflow.execute(
        inputs = {
            "PreprocessingJobName": preprocessing_job_name,
            "ToDoHPO": todoHPO,
            "ToDoTraining": todoTraining,
            "TrainingJobName": training_job_name,
            "TuningJobName": tuning_job_name,
            "ModelName": model_job_name,
            "EndpointName": endpoint_job_name,
            "TransformJobName": batch_transform_job_name
        }
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load parameters")
    parser.add_argument("--workflow-name", required=True)
    parser.add_argument("--workflow-execution-role", required=True)
    parser.add_argument("--processing-repository-uri", required=True)
    parser.add_argument("--require-hpo", required=True)
    parser.add_argument("--require-model-training", required=True)
    args = vars(parser.parse_args())
    print("args: {}".format(args))
    main(**args)
