from pyspark.sql import SparkSession
from pyspark.ml.regression import LinearRegression

# Create Spark Session
spark = SparkSession.builder.appName('Linear Regression').getOrCreate()
spark.sparkContext.setLogLevel('ERROR')

# Load training data
training = spark.read.format('libsvm').load('data/linear.txt')

lr = LinearRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)

# Fit the model
model = lr.fit(training)

# Print the coefficients and intercept for linear regression
print('\nLinear Regression:\n')
print('Coefficients: {}'.format(model.coefficients))
print('Intercept: {}'.format(model.intercept))

# Summarize the model over the training set and print out some metrics
sumary = model.summary
print()
print('NumIterations: {}'.format(sumary.totalIterations))
print('ObjectiveHistory: {}\n'.format(sumary.objectiveHistory))

sumary.residuals.show()

print('RMSE: {}'.format(sumary.rootMeanSquaredError))
print('r2: {}'.format(sumary.r2))