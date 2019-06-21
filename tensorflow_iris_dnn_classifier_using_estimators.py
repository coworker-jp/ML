#!/usr/bin/env python
# coding: utf-8

# # Creating, training, and serving using SageMaker Estimators
# 
# The **SageMaker Python SDK** helps you deploy your models for training and hosting in optimized, production ready containers in SageMaker. The SageMaker Python SDK is easy to use, modular, extensible and compatible with TensorFlow and MXNet. This tutorial focuses on **TensorFlow** and shows how we can train and host a TensorFlow DNNClassifier estimator in SageMaker using the Python SDK.
# 
# 
# TensorFlow's high-level machine learning API (tf.estimator) makes it easy to
# configure, train, and evaluate a variety of machine learning models.
# 
# 
# In this tutorial, you'll use tf.estimator to construct a
# [neural network](https://en.wikipedia.org/wiki/Artificial_neural_network)
# classifier and train it on the
# [Iris data set](https://en.wikipedia.org/wiki/Iris_flower_data_set) to
# predict flower species based on sepal/petal geometry. You'll write code to
# perform the following five steps:
# 
# 1.  Deploy a TensorFlow container in SageMaker
# 2.  Load CSVs containing Iris training/test data from a S3 bucket into a TensorFlow `Dataset`
# 3.  Construct a `tf.estimator.DNNClassifier` neural network classifier
# 4.  Train the model using the training data
# 5.  Host the model in an endpoint
# 6.  Classify new samples invoking the endpoint
# 
# This tutorial is a simplified version of TensorFlow's [get_started/estimator](https://www.tensorflow.org/get_started/estimator#fit_the_dnnclassifier_to_the_iris_training_data) tutorial **but using SageMaker and the SageMaker Python SDK** to simplify training and hosting.

# In[1]:


get_ipython().system('sudo ./setup.sh')


# ## The Iris dataset
# 
# The [Iris data set](https://en.wikipedia.org/wiki/Iris_flower_data_set) contains
# 150 rows of data, comprising 50 samples from each of three related Iris species:
# *Iris setosa*, *Iris virginica*, and *Iris versicolor*.
# 
# ![Petal geometry compared for three iris species: Iris setosa, Iris virginica, and Iris versicolor](https://www.tensorflow.org/images/iris_three_species.jpg) **From left to right,
# [*Iris setosa*](https://commons.wikimedia.org/w/index.php?curid=170298) (by
# [Radomil](https://commons.wikimedia.org/wiki/User:Radomil), CC BY-SA 3.0),
# [*Iris versicolor*](https://commons.wikimedia.org/w/index.php?curid=248095) (by
# [Dlanglois](https://commons.wikimedia.org/wiki/User:Dlanglois), CC BY-SA 3.0),
# and [*Iris virginica*](https://www.flickr.com/photos/33397993@N05/3352169862)
# (by [Frank Mayfield](https://www.flickr.com/photos/33397993@N05), CC BY-SA
# 2.0).**
# 
# Each row contains the following data for each flower sample:
# [sepal](https://en.wikipedia.org/wiki/Sepal) length, sepal width,
# [petal](https://en.wikipedia.org/wiki/Petal) length, petal width, and flower
# species. Flower species are represented as integers, with 0 denoting *Iris
# setosa*, 1 denoting *Iris versicolor*, and 2 denoting *Iris virginica*.
# 
# Sepal Length | Sepal Width | Petal Length | Petal Width | Species
# :----------- | :---------- | :----------- | :---------- | :-------
# 5.1          | 3.5         | 1.4          | 0.2         | 0
# 4.9          | 3.0         | 1.4          | 0.2         | 0
# 4.7          | 3.2         | 1.3          | 0.2         | 0
# &hellip;     | &hellip;    | &hellip;     | &hellip;    | &hellip;
# 7.0          | 3.2         | 4.7          | 1.4         | 1
# 6.4          | 3.2         | 4.5          | 1.5         | 1
# 6.9          | 3.1         | 4.9          | 1.5         | 1
# &hellip;     | &hellip;    | &hellip;     | &hellip;    | &hellip;
# 6.5          | 3.0         | 5.2          | 2.0         | 2
# 6.2          | 3.4         | 5.4          | 2.3         | 2
# 5.9          | 3.0         | 5.1          | 1.8         | 2
# 
# For this tutorial, the Iris data has been randomized and split into two separate
# CSVs:
# 
# *   A training set of 120 samples
#     iris_training.csv
# *   A test set of 30 samples
#     iris_test.csv
# 
# These files are provided in the SageMaker sample data bucket:
# **s3://sagemaker-sample-data-{region}/tensorflow/iris**. Copies of the bucket exist in each SageMaker region. When we access the data, we'll replace {region} with the AWS region the notebook is running in.

# ## Let us first initialize variables

# In[1]:


from sagemaker import get_execution_role
from sagemaker.session import Session

# S3 bucket for saving code and model artifacts.
# Feel free to specify a different bucket here if you wish.
bucket = Session().default_bucket()

# Location to save your custom code in tar.gz format.
custom_code_upload_location = 's3://{}/customcode/tensorflow_iris'.format(bucket)

# Location where results of model training are saved.
model_artifacts_location = 's3://{}/artifacts'.format(bucket)

#IAM execution role that gives SageMaker access to resources in your AWS account.
role = get_execution_role()


# # tf.estimator

# The tf.estimator framework makes it easy to construct and train machine learning models via its high-level Estimator API. Estimator offers classes you can instantiate to quickly configure common model types such as regressors and classifiers:
# 
# 
# *   **```tf.estimator.LinearClassifier```**:
#     Constructs a linear classification model.
# *   **```tf.estimator.LinearRegressor```**:
#     Constructs a linear regression model.
# *   **```tf.estimator.DNNClassifier```**:
#     Construct a neural network classification model.
# *   **```tf.estimator.DNNRegressor```**:
#     Construct a neural network regression model.
# *   **```tf.estimator.DNNLinearCombinedClassifier```**:
#     Construct a neural network and linear combined classification model.
# *   **```tf.estimator.DNNRegressor```**:
#     Construct a neural network and linear combined regression model.
#     
# More information about estimators can be found [here](https://www.tensorflow.org/extend/estimators)

# # Construct a deep neural network classifier

# ## Complete neural network source code 
# 
# Here is the full code for the neural network classifier:

# In[2]:


get_ipython().system('cat "iris_dnn_classifier.py"')


# With few lines of code, using SageMaker and TensorFlow, you can create a deep neural network model, ready for training and hosting. Let's give a deeper look at the code.

# ### Using a tf.estimator in SageMaker
# Using a TensorFlow estimator in SageMaker is very easy, you can create one with few lines of code:

# In[3]:


def estimator(model_path, hyperparameters):
    feature_columns = [tf.feature_column.numeric_column(INPUT_TENSOR_NAME, shape=[4])]
    return tf.estimator.DNNClassifier(feature_columns=feature_columns,
                                      hidden_units=[10, 20, 10],
                                      n_classes=3,
                                      model_dir=model_path)


# The code above first defines the model's feature columns, which specify the data
# type for the features in the data set. All the feature data is continuous, so
# `tf.feature_column.numeric_column` is the appropriate function to use to
# construct the feature columns. There are four features in the data set (sepal
# width, sepal height, petal width, and petal height), so accordingly `shape`
# must be set to `[4]` to hold all the data.
# 
# Then, the code creates a `DNNClassifier` model using the following arguments:
# 
# *   `feature_columns=feature_columns`. The set of feature columns defined above.
# *   `hidden_units=[10, 20, 10]`. Three
#     [hidden layers](http://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw),
#     containing 10, 20, and 10 neurons, respectively.
# *   `n_classes=3`. Three target classes, representing the three Iris species.
# *   `model_dir=model_path`. The directory in which TensorFlow will save
#     checkpoint data during model training. 

# ### Describe the training input pipeline
# 
# The `tf.estimator` API uses input functions, which create the TensorFlow
# operations that generate data for the model.
# We can use `tf.estimator.inputs.numpy_input_fn` to produce the input pipeline:

# In[5]:


def train_input_fn(training_dir, hyperparameters):
    training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
        filename=os.path.join(training_dir, 'iris_training.csv'),
        target_dtype=np.int,
        features_dtype=np.float32)

    return tf.estimator.inputs.numpy_input_fn(
        x={INPUT_TENSOR_NAME: np.array(training_set.data)},
        y=np.array(training_set.target),
        num_epochs=None,
        shuffle=True)()


# ### Describe the serving input pipeline:
# 
# After traininng your model, SageMaker will host it in a TensorFlow serving. You need to describe a serving input function:

# In[6]:


def serving_input_fn(hyperparameters):
    feature_spec = {INPUT_TENSOR_NAME: tf.FixedLenFeature(dtype=tf.float32, shape=[4])}
    return tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)()


# Now we are ready to submit the script for training.

# # Train a Model on Amazon SageMaker using TensorFlow custom code
# 
# We can use the SDK to run our local training script on SageMaker infrastructure.
# 
# 1. Pass the path to the iris_dnn_classifier.py file, which contains the functions for defining your estimator, to the sagemaker.TensorFlow init method.
# 2. Pass the S3 location that we uploaded our data to previously to the fit() method.

# In[4]:


from sagemaker.tensorflow import TensorFlow

iris_estimator = TensorFlow(entry_point='iris_dnn_classifier.py',
                            role=role,
                            framework_version='1.12.0',
                            output_path=model_artifacts_location,
                            code_location=custom_code_upload_location,
                            train_instance_count=1,
                            train_instance_type='ml.c4.xlarge',
                            training_steps=1000,
                            evaluation_steps=100)


# In[5]:


get_ipython().run_cell_magic('time', '', "import boto3\n\n# use the region-specific sample data bucket\nregion = boto3.Session().region_name\ntrain_data_location = 's3://sagemaker-sample-data-{}/tensorflow/iris'.format(region)\n\niris_estimator.fit(train_data_location)")


# In[ ]:





# # Deploy the trained Model  
# 
# The deploy() method creates an endpoint which serves prediction requests in real-time.

# In[6]:


get_ipython().run_cell_magic('time', '', "iris_predictor = iris_estimator.deploy(initial_instance_count=1,\n                                       instance_type='ml.t2.medium')")


# #  Invoke the Endpoint to get inferences

# Invoking prediction:

# In[9]:


iris_predictor.predict([6.4, 4.2, 4.5, 1.5]) #expected label to be 1


# # (Optional) Delete the Endpoint
# 
# After you have finished with this example, remember to delete the prediction endpoint to release the instance(s) associated with it.

# In[10]:


print(iris_predictor.endpoint)


# In[11]:


import sagemaker

sagemaker.Session().delete_endpoint(iris_predictor.endpoint)


# In[ ]:





# In[ ]:




