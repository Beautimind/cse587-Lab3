from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark import SparkConf, SparkContext
from operator import add
from pyspark.sql import SparkSession

sc =SparkContext()
spark = SparkSession(sc)
sentenceData = spark.createDataFrame([
    (0.0, "Hi I heard about Spark"),
    (0.0, "I wish Java could use case classes"),
    (1.0, "Logistic regression models are neat")
], ["label", "sentence"])

tokenizer = Tokenizer(inputCol="sentence", outputCol="words")
wordsData = tokenizer.transform(sentenceData)
wordsData.show()
hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=20)
featurizedData = hashingTF.transform(wordsData)
# alternatively, CountVectorizer can also be used to get term frequency vectors
featurizedData.show()
idf = IDF(inputCol="rawFeatures", outputCol="features")
idfModel = idf.fit(featurizedData)
rescaledData = idfModel.transform(featurizedData)

rescaledData.select("label", "rawFeatures","features").show(1,False)

politics_path="politics.csv"
sports_path="sports.csv"
business_path="business.csv"
science_path="science.csv"

unknownpolitics="unknownpolitics.csv"
unknownsports="unknownsports.csv"
unknownbusiness="unknownbusiness.csv"
unknownscience="unknownscience.csv"

from pyspark.sql import SQLContext
from pyspark.sql.functions import lit
from pyspark.sql.functions import monotonically_increasing_id 
sqlContext = SQLContext(sc)

politics = sqlContext.read.format('com.databricks.spark.csv').options(header='true', inferschema='true').load(politics_path)
sports = sqlContext.read.format('com.databricks.spark.csv').options(header='true', inferschema='true').load(sports_path)
business = sqlContext.read.format('com.databricks.spark.csv').options(header='true', inferschema='true').load(business_path)
science = sqlContext.read.format('com.databricks.spark.csv').options(header='true', inferschema='true').load(science_path)

politics=politics.drop("headline").withColumn("type",lit(0))#.withColumn("ID",monotonically_increasing_id())
politics.show()
sports=sports.drop("headline").withColumn("type",lit(1))
sports.show()
business=business.drop("headline").withColumn("type",lit(2))
business.show()
science=science.drop("headline").withColumn("type",lit(3))
science.show()

unpolitics = sqlContext.read.format('com.databricks.spark.csv').options(header='true', inferschema='true').load(unknownpolitics)
unsports = sqlContext.read.format('com.databricks.spark.csv').options(header='true', inferschema='true').load(unknownsports)
unbusiness = sqlContext.read.format('com.databricks.spark.csv').options(header='true', inferschema='true').load(unknownbusiness)
unscience = sqlContext.read.format('com.databricks.spark.csv').options(header='true', inferschema='true').load(unknownscience)

unpolitics=unpolitics.drop("headline").withColumn("type",lit(0))#.withColumn("ID",monotonically_increasing_id())
unpolitics.show()
unsports=unsports.drop("headline").withColumn("type",lit(1))
unsports.show()
unbusiness=unbusiness.drop("headline").withColumn("type",lit(2))
unbusiness.show()
unscience=unscience.drop("headline").withColumn("type",lit(3))
unscience.show()

unall=unpolitics.union(unsports).union(unbusiness).union(unscience)

alldata=politics.union(sports).union(business).union(science)
alldata=alldata.withColumn("ID",monotonically_increasing_id())
alldata.show()

train_test=alldata.randomSplit([0.8,0.2])
traindata=train_test[0]
traindata.show()
# print "the number of train data is ",traindata.count()
testdata=train_test[1]
testdata.show()
# print "the number of test data is ",testdata.count()

# tokenize the word
from pyspark.ml.feature import RegexTokenizer
regexTokenizer = RegexTokenizer(inputCol="story-content", outputCol="words", pattern="\\W+")
regexTokenized = regexTokenizer.transform(traindata)
regexTokenized.show()

#remove stop words
from pyspark.ml.feature import StopWordsRemover
remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
removed=remover.transform(regexTokenized)

#count TF
from pyspark.ml.feature import CountVectorizer
hashingTF = HashingTF(inputCol="filtered_words", outputCol="rawFeatures", numFeatures=800)
featurizedData =hashingTF.transform(removed)

#count TF-IDF
idf = IDF(inputCol="rawFeatures", outputCol="features")
idfModel = idf.fit(featurizedData)
rescaledData = idfModel.transform(featurizedData)
rescaledData.select("type","rawFeatures", "features").show(1,False)

from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
##random forest
rf = RandomForestClassifier(labelCol="type", featuresCol="features", numTrees=10)

pipeline = Pipeline(stages=[regexTokenizer, remover, hashingTF, idfModel,rf])
model=pipeline.fit(traindata)

predictions = model.transform(testdata)

evaluator = MulticlassClassificationEvaluator(
    labelCol="type", predictionCol="prediction", metricName="accuracy")
accuracy1 = evaluator.evaluate(predictions)
# print("Test Error for random forest is = %g" % (1.0 - accuracy))

#test for logistic regression
from pyspark.ml.classification import LogisticRegression
lr = LogisticRegression(maxIter=50, regParam=0.3, elasticNetParam=0,labelCol="type", featuresCol="features")
pipeline2=pipeline = Pipeline(stages=[regexTokenizer, remover, hashingTF, idfModel,lr])
lrmodel=pipeline2.fit(traindata)
predictions=lrmodel.transform(testdata)
evaluator = MulticlassClassificationEvaluator(
    labelCol="type", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Test Error for random forest is = %g" % (1.0 - accuracy1))
print("Test Error for logistic regression is = %g" % (1.0 - accuracy))


#prediction for unknowndatra
unprediction1=model.transform(unall)
unaccuracy1=evaluator.evaluate(unprediction1)
unprediction2=lrmodel.transform(unall)
unaccuracy2=evaluator.evaluate(unprediction2)
print("Test Error for random forest is = %g" % (1.0 - accuracy1))
print("Test Error for logistic regression is = %g" % (1.0 - accuracy))
print("Unknown set Error for random forest is = %g" % (1.0 - unaccuracy1))
print("Unknown set Error for logistic regression is = %g" % (1.0 - unaccuracy2))