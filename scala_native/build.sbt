scalaVersion := "2.13.7"
enablePlugins(ScalaNativePlugin)

name := "scalapy_torch_example"
version := "1.0"

libraryDependencies += "me.shadaj" %%% "scalapy-core" % "0.5.1"
libraryDependencies += "org.scala-native" %%% "scalalib" % "0.4.2"

fork := true
lazy val pythonLdFlags: Seq[String] = Seq("-L/usr/local/Caskroom/miniconda/base/envs/cv38/lib/python3.8/config-3.8-darwin",
  "-lpython3.8", "-ldl", "-framework", "CoreFoundation")

nativeLinkingOptions ++= pythonLdFlags
