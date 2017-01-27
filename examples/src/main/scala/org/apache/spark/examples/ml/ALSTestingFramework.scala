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

object ALSTestingFramework {


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

  val map = new util.HashMap[String, PrintStream]()

  var bestAlsParams = AlsParams(Double.PositiveInfinity, 0, 0, "", "", "", "")

  def main(args: Array[String]) {
    val spark = SparkSession
      .builder
      .appName("ALSTestingFramework")
      .getOrCreate()
    import spark.implicits._

    // $example on$
    val ratings = spark.read.textFile("data/mllib/als/sample_movielens_ratings.txt")
      .map(parseRating)
      .toDF()
    val Array(training, test) = ratings.randomSplit(Array(0.8, 0.2))

    val maxIterations = Array(5, 10, 20)
    val lambdasForVariableUpdater = Array(1, 5, 10, 15)
    val lambdasForConstUpdater = Array(0.1, 0.5, 1)
    val ranks = Array(8, 12)
    val divisions = Array(2, 4)
    val solverClasses = Array(CholeskySolver, NNLSSolver,
      NNLSSolverDifferentLambda)

    val outDir = args(0)

    new File(outDir).mkdirs()

    for (solver1 <- solverClasses; solver2 <- solverClasses) {
      val fileName = solver1().getClass.getSimpleName + "_" + solver2().getClass.getSimpleName
      val file = new File(outDir + fileName)
      val outputStream = new PrintStream(file)
      map.put(fileName, outputStream)
    }

    for (solver1 <- solverClasses; solver2 <- solverClasses;
         maxIter <- maxIterations; rank <- ranks) {
      for (constLambda <- lambdasForConstUpdater) {
        runWithLambdaUpdater(ConstLambda(constLambda), ConstLambda(constLambda),
          solver1(),
          solver2(),
          training, test, maxIter, rank)

        for (variableLambda <- lambdasForVariableUpdater) {
          for (division <- divisions) {
            runWithLambdaUpdater(
              VariableLambda(variableLambda, Math.pow(10, -10.0), division),
              ConstLambda(constLambda),
              solver1(),
              solver2(),
              training, test, maxIter, rank)

            runWithLambdaUpdater(
              ConstLambda(constLambda),
              VariableLambda(variableLambda, Math.pow(10, -10.0), division),
              solver1(),
              solver2(),
              training, test, maxIter, rank)
          }
        }
      }
    }

    val file = new File(outDir + "theBest")
    val outputStream = new PrintStream(file)
    outputStream.println(bestAlsParams.toString)
    outputStream.close()


    for (solver1 <- solverClasses; solver2 <- solverClasses) {
      val fileName = solver1().getClass.getSimpleName + "_" + solver2().getClass.getSimpleName
      map.get(fileName).close()
    }

    spark.stop()
  }


  def runWithLambdaUpdater(lambdaUpdater1: LambdaUpdater, lambdaUpdater2: LambdaUpdater,
                           solver1: LeastSquaresNESolver, solver2: LeastSquaresNESolver,
                           training: Dataset[_], test: Dataset[_],
                           maxIter: Int, rank: Int) {
    val als = new ALS()
      .setMaxIter(maxIter)
      .setRank(rank)
      .setUserCol("userId")
      .setItemCol("movieId")
      .setRatingCol("rating")

    val model = als.fit(training, lambdaUpdater1, lambdaUpdater2,
      solver1, solver2)

    val predictions = model.transform(test)
    val evaluator = new RegressionEvaluator()
      .setMetricName("rmse")
      .setLabelCol("rating")
      .setPredictionCol("prediction")

    val rmse = evaluator.evaluate(predictions)

    if (rmse < bestAlsParams.rmse) {
      bestAlsParams = AlsParams(rmse, maxIter, rank,
        solver1.getClass.getSimpleName,
        solver2.getClass.getSimpleName,
        lambdaUpdater1.toString, lambdaUpdater2.toString)
    }

    val fileName = solver1.getClass.getSimpleName + "_" + solver2.getClass.getSimpleName

    val csvLine = "%015.10f;%s;%s;%s;%s;%d;%d".format(rmse, solver1.getClass.getSimpleName,
      solver2.getClass.getSimpleName, lambdaUpdater1.toString, lambdaUpdater2.toString,
      maxIter, rank)

    map.get(fileName).println(csvLine)

  }

}
