package fr.esgi.spark


import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer, VectorAssembler}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.functions.round
import org.apache.spark.sql.types.{DoubleType, IntegerType}

object SparkML {

  def main(args: Array[String]): Unit = {

    //Instancier le spark session
    val spark = SparkSession.builder()
      .appName("Spark SQL TD1")
      .config("spark.driver.memory","512m")
      .master("local[8]")
      .getOrCreate()
    import spark.implicits._

    val data_gender = spark.read
      .option("inferSchema", "true")
      .option("header", "true")
      .option("delimiter",",")
      .format("csv")
      .load("../../../../resources/gender_submission.csv")

     data_gender.show()


    val rawTraining = spark.read
      .option("inferSchema", "true")
      .option("header", "true")
      .option("delimiter",",")
      .format("csv")
      .load("../../../../resources/train.csv")

    val sizeTraining = rawTraining.select("PassengerId").count()
    print("il y a :", sizeTraining)


    val rawTest = spark.read
      .option("inferSchema", "true")
      .option("header", "true")
      .option("delimiter",",")
      .format("csv")
      .load("../../../../resources/test.csv")

    rawTest.show()
    // PARTIE 1
    def calcMeanAge(df: DataFrame, inputCol: String): Double = df
      .agg(avg(col(inputCol)))
      .head
      .getDouble(0)

    def fillMissingAge(df: DataFrame, inputCol: String, outputCol: String, replacementValue:
    Double): DataFrame = {
      val ageOrMeanAge: (Any) => Double = age => age match {
        case age: Double => age
        case _ => replacementValue
      }
      val udfAgeOrMeanAge = udf(ageOrMeanAge)
      df.withColumn(outputCol, udfAgeOrMeanAge(col(inputCol)))
    }



    val mean_age = calcMeanAge(rawTraining,"Age")
    // print("mean age :", mean_age)

    val training = fillMissingAge(rawTraining,"Age","Age_cleaned",calcMeanAge(rawTraining,"Age"))
    // training.show()

    val test = fillMissingAge(rawTest,"Age","Age_cleaned",calcMeanAge(rawTest,"Age"))
    // test.show()

    // part 2

    val labelIndexerModel = new StringIndexer()
      .setInputCol("Survived")
      .setOutputCol("label")
      .fit(training)

    val trainingWithLabel = labelIndexerModel.transform(training)

    // trainingWithLabel.show()

    val SexIndexerModel = new StringIndexer()
      .setInputCol("Sex")
      .setOutputCol("Sex_indexed")



    val vectorizedDf = new VectorAssembler()
      .setInputCols(
        Array("Pclass","Sex_indexed","Age_cleaned")
      )
      .setOutputCol("features")
      .transform(SexIndexerModel.fit(trainingWithLabel).transform(trainingWithLabel))

    // vectorizedDf.show()


    val LabelIndexer = new StringIndexer()
      .setInputCol("Survived")
      .setOutputCol("label")

    val SexIndexer = new StringIndexer()
      .setInputCol("Sex")
      .setOutputCol("Sex_indexed")

    val vectorAssembler = new VectorAssembler()
      .setInputCols( Array("Pclass","Sex_indexed","Age_cleaned"))
      .setOutputCol("features")

    val randomForest = new RandomForestClassifier()

    val pipeline = new Pipeline()
      .setStages(
        Array(
          LabelIndexer,
          SexIndexer,
          vectorAssembler,
          randomForest
        )
      )

    val model = pipeline.fit(training)
    val predictions = model.transform(test)



    val bilan = predictions.join(
      data_gender,
      Seq("PassengerId"),
      "inner")



    bilan
      .groupBy(col("Survived"),col("prediction"))
      .agg(count("PassengerId"))
      .show()


    val x =bilan.filter(
                (col("Survived")===1 && col("prediction")===1)
                  ||
                (col("Survived")===0 && col("prediction")===0))
        .agg(count("PassengerId"))
        .head()
        .getLong(0)



    val y = bilan.agg(count("PassengerId"))
      .head
      .getLong(0)

    val accuracy = x.toFloat / y.toFloat

    println("l'accuracy est de : " + accuracy*100 + " %")


    val paramGrid = new ParamGridBuilder()
      .addGrid(randomForest.maxDepth,Array(2, 5, 10))
      .addGrid(randomForest.numTrees,Array(15, 30, 50))
      .addGrid(randomForest.maxBins,Array(16, 32, 64))
      .build()


    val MulticlassEvaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")

    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")


    val crossValidator = new CrossValidator()
      .setNumFolds(3)
      .setEstimator(pipeline)
      .setEstimatorParamMaps(paramGrid)
      .setEvaluator(evaluator)

    val cvModel = crossValidator.fit(training)

    val cvResult = cvModel.transform(test)

    val crossValidBilan = cvResult.join(data_gender,"PassengerId")

    crossValidBilan
      .groupBy(col("Survived"),col("prediction"))
      .agg(count("PassengerId"))
      .show()

    val crossValid_x = crossValidBilan.filter(
      (col("Survived")===1 && col("prediction")===1)
        ||
        (col("Survived")===0 && col("prediction")===0))
      .agg(count("PassengerId"))
      .head()
      .getLong(0)



    val crossValid_y = crossValidBilan.agg(count("PassengerId"))
      .head
      .getLong(0)

    val crossValid_accuracy = crossValid_x.toFloat / crossValid_y.toFloat

    println("l'accuracy du CrossValidation est de : " + crossValid_accuracy*100 + " %")

    /*

 ---> ML
+--------+----------+------------------+
|Survived|prediction|count(PassengerId)|
+--------+----------+------------------+
|       1|       0.0|                46|
|       0|       0.0|               262|
|       1|       1.0|               106|
|       0|       1.0|                 4|
+--------+----------+------------------+
l'accuracy est de : 88.03828 %

--> CrossValidation
+--------+----------+------------------+
|Survived|prediction|count(PassengerId)|
+--------+----------+------------------+
|       1|       0.0|                42|
|       0|       0.0|               259|
|       1|       1.0|               110|
|       0|       1.0|                 7|
+--------+----------+------------------+
l'accuracy du CrossValidation est de : 88.27751 %


 Le resultat de cross validation est infinement meilleur que celui sans la cross validation .

  Pourquoi ? :
  Nous avons essayé plusieurs paramètres pour que notre modèle trouve le meilleur

     */

  }
}