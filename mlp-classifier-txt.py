from pyspark.sql import SparkSession
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Create Spark Session
spark = SparkSession.builder.appName('Multi Layer Perceptron Classifier').getOrCreate()
spark.sparkContext.setLogLevel('ERROR')

# Load training data
data = spark.read.format('libsvm').load('data/mlp.txt')

# Split the data into train and test
splits = data.randomSplit([0.6, 0.4], 1234)
train = splits[0]
test = splits[1]

# specify layers for the neural network:
# input layer of size 4 (features), two intermediate of size 5 and 4
# and output of size 3 (classes)
layers = [4, 5, 4, 3]

# create the trainer and set its parameters
trainer = MultilayerPerceptronClassifier(maxIter=100, layers=layers, blockSize=128, seed=1234)

# train the model
model = trainer.fit(train)

# compute accuracy on the test set
result = model.transform(test)
predictionAndLabels = result.select('prediction', 'label')
evaluator = MulticlassClassificationEvaluator(metricName='accuracy')

print('\nMulti Layer Perceptron Classifier:')
print('Test set accuracy = {}'.format(evaluator.evaluate(predictionAndLabels)))