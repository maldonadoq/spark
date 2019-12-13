from pyspark.sql import SparkSession
from pyspark.ml.classification import MultilayerPerceptronClassificationModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Create Spark Session
spark = SparkSession.builder.appName('Multi Layer Perceptron Classifier').getOrCreate()
spark.sparkContext.setLogLevel('ERROR')

# Load training data
data = spark.read.format('libsvm').load('data/mnist_test.txt')

# load the model
model = MultilayerPerceptronClassificationModel.load('model/mnist/')

# compute accuracy on the test set
result = model.transform(data)
predictionAndLabels = result.select('prediction', 'label')

print('\nMulti Layer Perceptron Classifier:')
predictionAndLabels.show()