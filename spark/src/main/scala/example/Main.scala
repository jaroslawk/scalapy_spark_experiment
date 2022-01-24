package example

import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.SeqConverters
import org.apache.spark.sql.SparkSession

object ScalaExampleMain {

  def main(args: Array[String]) {
    val spark =
      SparkSession.builder()
        .appName("Dataset-Basic")
        .master("local[4]")
        .getOrCreate()

    import spark.implicits._
    val tuples = Seq((1, "one", "un"),
      (2, "two", "deux"),
      (3, "three", "trois"))
    val mlInput = tuples
      .toDS()
      .where($"_1" > 0).map { case (number, valOne, valTwo) => (number, valOne) }
      .collect()
      .toSeq
      .toPythonCopy

    execute("fancy_ml.py")

    val res = py.Dynamic.global.fancy_ml(mlInput).as[Seq[String]]
    println(res)
  }
}