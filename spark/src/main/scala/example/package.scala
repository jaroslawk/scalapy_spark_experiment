import me.shadaj.scalapy.interpreter.CPythonInterpreter

import scala.io.Source

package object example {
  def execute(pyFile: String) = {
    val fancy_python = Source.fromResource(pyFile).getLines.mkString("\n")
    CPythonInterpreter.execManyLines(fancy_python)
  }
}