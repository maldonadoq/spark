from pyspark.sql import SparkSession
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Create Spark Session
spark = SparkSession.builder.appName('Multi Layer Perceptron Classifier').getOrCreate()
spark.sparkContext.setLogLevel('ERROR')

# Load training and Testing data
data_train = spark.read.format('libsvm').load('data/mnist_train.txt')
data_test = spark.read.format('libsvm').load('data/mnist_test.txt')

# specify layers for the neural network:
# input layer of size 784 (features), two intermediate of size 512 and 256
# and output of size 10 (classes)
layers = [784, 512, 256, 10]
epochs = 250

# create the trainer and set its parameters
trainer = MultilayerPerceptronClassifier(maxIter=epochs, layers=layers)

# train the model
model = trainer.fit(data_train)
model.write().overwrite().save('model/mnist/')

# compute accuracy on the test set
result = model.transform(data_test)
predictionAndLabels = result.select('prediction', 'label')
evaluator = MulticlassClassificationEvaluator(metricName='accuracy')

print('\nMulti Layer Perceptron Classifier:')
print('Test set accuracy = {}'.format(evaluator.evaluate(predictionAndLabels)))