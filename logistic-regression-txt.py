from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression

# Create Spark Session
spark = SparkSession.builder.appName('Logistic Regression').getOrCreate()
spark.sparkContext.setLogLevel('ERROR')

# Load training data
training = spark.read.format('libsvm').load('data/logistic.txt')

lr = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)

# Fit the model
model = lr.fit(training)

# Print the coefficients and intercept for logistic regression
print('\nLogistic Regression:\n')
print('Coefficients: {}'.format(model.coefficients))
print('Intercept: {}'.format(model.intercept))

# We can also use the multinomial family for binary classification
mlr = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8, family='multinomial')

# Fit the model
mmodel = mlr.fit(training)

# Print the coefficients and intercepts for logistic regression with multinomial family
print('\n')
print('Multinomial coefficients: {}'.format(mmodel.coefficientMatrix))
print('Multinomial intercepts: {}'.format(mmodel.interceptVector))