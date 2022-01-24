import me.shadaj.scalapy.interpreter.{CPythonInterpreter, Platform, PyValue}
import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.ConvertableToSeqElem
import me.shadaj.scalapy.readwrite.Writer

import java.util.NoSuchElementException
import scala.collection.Iterator
import scala.io.Source

package object example {

  def execute(pyFile: String) = {
    val fancy_python = Source.fromResource(pyFile).getLines.mkString("\n")
    CPythonInterpreter.execManyLines(fancy_python)
  }

  // TODO: fixme
  implicit def writableSeqElem[Double](implicit writer: Writer[Double]): ConvertableToSeqElem[Double] = new ConvertableToSeqElem[Double] {
    def convertCopy(v: Double): Platform.Pointer = writer.writeNative(v)

    def convertProxy(v: Double): PyValue = writer.write(v)
  }

  def len(pyVal: py.Dynamic) = py.Dynamic.global.len(pyVal).as[Int]
  def del(pyVal: py.Dynamic) = py.Dynamic.global.del(pyVal)

  // TODO:fixme
  def maybeValue(py_it: py.Dynamic): Option[py.Dynamic] = {
    try {
      Some(py_it.__next__())
    }
    catch {
      case e: Exception => None
    }
  }

  // TODO:fixme
  def toTraversable(py_it: py.Dynamic): Iterator[py.Dynamic] = new Iterator[py.Dynamic] {
    var maybeDynamic: Option[py.Dynamic] = maybeValue(py_it)

    override def hasNext: Boolean = maybeDynamic.nonEmpty

    override def next(): py.Dynamic = {
      val curr = maybeDynamic
      maybeDynamic = maybeValue(py_it)
      curr.get
    }
  }

  def timeIt[R](block: => R): R = {
    val t0 = System.nanoTime()
    val result = block
    val t1 = System.nanoTime()
    println("Elapsed time: " + (t1 - t0) / 1000000000 + "s")
    result
  }
}
