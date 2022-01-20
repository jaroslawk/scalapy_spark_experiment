name := "MyProject"  
version := "1.0"  
scalaVersion := "2.12.10"

libraryDependencies += "org.apache.hadoop" % "hadoop-common" % "3.2.0"
libraryDependencies += "org.apache.hadoop" % "hadoop-hdfs-client" % "3.2.0"
libraryDependencies += "org.apache.spark" %% "spark-core" % "3.0.0"
libraryDependencies += "org.apache.spark" %% "spark-sql" % "3.0.0"

libraryDependencies += "me.shadaj" %% "scalapy-core" % "0.5.1"

Compile / unmanagedResourceDirectories += (Compile / sourceDirectory).value / "python"

fork := true

javaOptions += s"-Djna.library.path=/usr/local/Caskroom/miniconda/base/envs/cv38/lib"

