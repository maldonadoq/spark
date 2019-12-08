from pyspark.sql import SparkSession
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

# Create Spark Session
spark = SparkSession.builder.appName('Kmeans Clustering').getOrCreate()
spark.sparkContext.setLogLevel('ERROR')

# Loads data.
dataset = spark.read.format('libsvm').load('data/kmeans.txt')

# Trains a k-means model.
kmeans = KMeans().setK(2).setSeed(1)
model = kmeans.fit(dataset)

# Make predictions
predictions = model.transform(dataset)

# Evaluate clustering by computing Silhouette score
evaluator = ClusteringEvaluator()
silhouette = evaluator.evaluate(predictions)
print('\nKmeans Clustering:\n')
print('Silhouette with squared euclidean distance = {}'.format(silhouette))

# Shows the result.
centers = model.clusterCenters()
print('Cluster Centers: ')
for center in centers:
    print('  ', center) 