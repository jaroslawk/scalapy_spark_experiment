scalaVersion := "3.0.2"

name := "scalapy_graph_example"
version := "1.0"

libraryDependencies += "org.scalactic" %% "scalactic" % "3.2.10"
libraryDependencies += "org.scalatest" %% "scalatest" % "3.2.10" % "test"
libraryDependencies += "org.scalatest" %% "scalatest-flatspec" % "3.2.10" % "test"
libraryDependencies += "me.shadaj" %% "scalapy-core" % "0.5.1"

fork := true

Compile / unmanagedResourceDirectories += (Compile / sourceDirectory).value / "python"

javaOptions += s"-Djna.library.path=/usr/local/Caskroom/miniconda/base/envs/cv38/lib"

