from pyspark.sql import SparkSession
from pyspark.ml.regression import LinearRegression

from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler

# Create Spark Session
spark = SparkSession.builder.appName('Linear Regression').getOrCreate()
spark.sparkContext.setLogLevel('ERROR')

# Read CSV
dataset = spark.read.csv('data/linear.csv', inferSchema=True, header=True)

print('\nLinear Regression:\n')
# Show first 5 rows of dataset 
dataset.show(5)
# Show Column Names
dataset.printSchema()

# Four Variables
featureassembler = VectorAssembler(inputCols = ["Avg Session Length","Time on App","Time on Website","Length of Membership"], outputCol = "Independent Features")

# Independent Features
output = featureassembler.transform(dataset)

output.show(5)
output.select("Independent Features").show(5)
output.columns

# Data in matrix [X1, X2, X3, X4][Y]
finalized_data = output.select("Independent Features","Yearly Amount Spent")
finalized_data.show(5)

# Split Training an Testing Data
train_data, test_data = finalized_data.randomSplit([0.75,0.25])

# Linear Regression Instance [X,Y]
regressor = LinearRegression(featuresCol='Independent Features', labelCol='Yearly Amount Spent')
# Training
regressor = regressor.fit(train_data)

# Get coeficient of Linear Regression
print('Regressor Coefficients')
print(regressor.coefficients)

# Predictions
pred_results = regressor.evaluate(test_data)
pred_results.predictions.show(10)