/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// scalastyle:off println
package org.apache.spark.examples.ml

// $example on$
import java.io.{File, PrintStream}
import java.util

import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.recommendation._
import org.apache.spark.sql.Dataset


// $example off$
import org.apache.spark.sql.SparkSession

object ALSTestingIters {


  // $example on$
  case class Rating(userId: Int, movieId: Int, rating: Float, timestamp: Long)

  def parseRating(str: String): Rating = {
    val fields = str.split("::")
    assert(fields.size == 4)
    Rating(fields(0).toInt, fields(1).toInt, fields(2).toFloat, fields(3).toLong)
  }

  // $example off$

  case class AlsParams(rmse: Double, maxIter: Int, rank: Int, solver1: String, solver2: String,
                       lambdaUpdater1: String, lambdaUpdater2: String)

  // val map = new util.HashMap[String, PrintStream]()

  var outputDir : String = ""

  //  var bestAlsParams = AlsParams(Double.PositiveInfinity, 0, 0, "", "", "", "")

  def main(args: Array[String]) {
    val spark = SparkSession
      .builder
      .appName("ALSTestingFramework")
      .getOrCreate()
    import spark.implicits._

    val outDir = args(0)
    val inputFile = args(1)
    val maxIterParams = args(2)
    val lambdasForVariableUpdaterParams = args(3)
    val lambdasForConstUpdaterParams = args(4)
    val ranksParams = args(5)
    val divisionsParam = args(6)
    val minLambdaInVariableUpdaterParam = args(7)
    val solver1Params = args(8)
    val solver2Params = args(9)

    // $example on$
    //    val ratings = spark.read.textFile("data/mllib/als/sample_movielens_ratings.txt")
    //      .map(parseRating)
    //      .toDF()

    val ratings = spark.read.textFile(inputFile)
      .map(parseRating)
      .toDF()
    val Array(training, test) = ratings.randomSplit(Array(0.8, 0.2))

    val maxIter = maxIterParams.toInt
    val lambdasForVariableUpdater = lambdasForVariableUpdaterParams.split(",").map(_.toDouble)
    val lambdasForConstUpdater = lambdasForConstUpdaterParams.split(",").map(_.toDouble)
    val ranks = ranksParams.split(",").map(_.toInt)
    val divisions = divisionsParam.split(",").map(_.toDouble)
    val minLambdaInVariableUpdater = minLambdaInVariableUpdaterParam.toDouble

    val solvers = Array(CholeskySolver, NNLSSolver, NNLSSolverDifferentLambda)

    val solverClasses1 = solver1Params.split(",").map(_.toInt).map(index => solvers(index))
    val solverClasses2 = solver2Params.split(",").map(_.toInt).map(index => solvers(index))


    new File(outDir).mkdirs()
    outputDir = outDir

//    for (solver1 <- solverClasses1; solver2 <- solverClasses2) {
//      val fileName = solver1().getClass.getSimpleName + "_" + solver2().getClass.getSimpleName
//      val file = new File(outDir + fileName)
//      val outputStream = new PrintStream(file)
//      map.put(fileName, outputStream)
//    }

    for (solver1 <- solverClasses1; solver2 <- solverClasses2;
         rank <- ranks) {
      for (constLambda <- lambdasForConstUpdater) {
        runWithLambdaUpdater(ConstLambda(constLambda), ConstLambda(constLambda),
          solver1(),
          solver2(),
          training, test, maxIter, rank)

        for (variableLambda <- lambdasForVariableUpdater) {
          for (division <- divisions) {
            runWithLambdaUpdater(
              VariableLambda(variableLambda, minLambdaInVariableUpdater, division),
              ConstLambda(constLambda),
              solver1(),
              solver2(),
              training, test, maxIter, rank)

            runWithLambdaUpdater(
              ConstLambda(constLambda),
              VariableLambda(variableLambda, minLambdaInVariableUpdater, division),
              solver1(),
              solver2(),
              training, test, maxIter, rank)
          }
        }
      }
    }

    //    val file = new File(outDir + "theBest")
    //    val outputStream = new PrintStream(file)
    //    outputStream.println(bestAlsParams.toString)
    //    outputStream.close()

//    for (solver1 <- solverClasses1; solver2 <- solverClasses2) {
//      val fileName = solver1().getClass.getSimpleName + "_" + solver2().getClass.getSimpleName
//      map.get(fileName).close()
//    }

    spark.stop()
  }


  def runWithLambdaUpdater(lambdaUpdater1: LambdaUpdater, lambdaUpdater2: LambdaUpdater,
                           solver1: LeastSquaresNESolver, solver2: LeastSquaresNESolver,
                           training: Dataset[_], test: Dataset[_],
                           maxIter: Int, rank: Int) {
    val start = System.currentTimeMillis()
    val als = new ALS()
      .setMaxIter(maxIter)
      .setRank(rank)
      .setUserCol("userId")
      .setItemCol("movieId")
      .setRatingCol("rating")

    val modelList = als.fit_list(training, lambdaUpdater1, lambdaUpdater2, solver1, solver2)

    val iterRmse = modelList.map {
      case (i: Int, model: ALSModel) =>
        val predictions = model.transform(test)
        val evaluator = new RegressionEvaluator()
          .setMetricName("rmse")
          .setLabelCol("rating")
          .setPredictionCol("prediction")

        val rmse = evaluator.evaluate(predictions)

        model.itemFactors.unpersist()
        model.userFactors.unpersist()

        (i, rmse)
    }

    //    val model = als.fit(training, lambdaUpdater1, lambdaUpdater2,
    //      solver1, solver2)
    //
    //    val predictions = model.transform(test)
    //    val evaluator = new RegressionEvaluator()
    //      .setMetricName("rmse")
    //      .setLabelCol("rating")
    //      .setPredictionCol("prediction")
    //
    //    val rmse = evaluator.evaluate(predictions)

    //    model.itemFactors.unpersist()
    //    model.userFactors.unpersist()
    //
    //    if (rmse < bestAlsParams.rmse) {
    //      bestAlsParams = AlsParams(rmse, maxIter, rank,
    //        solver1.getClass.getSimpleName,
    //        solver2.getClass.getSimpleName,
    //        lambdaUpdater1.toString, lambdaUpdater2.toString)
    //    }

    val durration = System.currentTimeMillis() - start

    val fileName = solver1.getClass.getSimpleName + "_" + solver2.getClass.getSimpleName +
      "_" + lambdaUpdater1.filename() + "_" + lambdaUpdater2.filename() + s"_$rank"
    val file = new File(outputDir + fileName)
    val outputStream = new PrintStream(file)

    iterRmse.foreach {
      case (i: Int, rmse: Double) =>
        val csvLine = "%015.10f;%s;%s;%s;%s;%d;%d".format(rmse, solver1.getClass.getSimpleName,
          solver2.getClass.getSimpleName, lambdaUpdater1.toString, lambdaUpdater2.toString,
          i, rank)

       outputStream.println(csvLine)
    }

    outputStream.close()

    val performanceFile = new File(outputDir + fileName + "-time")
    val performanceOutStream = new PrintStream(performanceFile)
    performanceOutStream.println( durration )
    performanceOutStream.close()
  }

}
