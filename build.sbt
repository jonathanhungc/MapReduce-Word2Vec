ThisBuild / version := "0.1.0-SNAPSHOT"

ThisBuild / scalaVersion := "2.13.12"

lazy val root = (project in file("."))
  .settings(
    name := "homework1"
  )

// https://mvnrepository.com/artifact/org.apache.hadoop/hadoop-common
libraryDependencies += "org.apache.hadoop" % "hadoop-common" % "3.4.0"
// https://mvnrepository.com/artifact/org.apache.hadoop/hadoop-mapreduce-client-core
libraryDependencies += "org.apache.hadoop" % "hadoop-mapreduce-client-core" % "3.4.0"
libraryDependencies += "org.apache.hadoop" % "hadoop-mapreduce-client-jobclient" % "3.4.0"

libraryDependencies += "com.knuddels" % "jtokkit" % "1.1.0"

libraryDependencies += "org.slf4j" % "slf4j-api" % "2.0.16"

libraryDependencies += "ch.qos.logback" % "logback-classic" % "1.5.6"

libraryDependencies += "com.typesafe" % "config" % "1.4.3"

libraryDependencies += "org.deeplearning4j" % "deeplearning4j-core" % "1.0.0-M2.1"

libraryDependencies += "org.deeplearning4j" % "deeplearning4j-ui-model" % "1.0.0-M2.1"

libraryDependencies += "org.deeplearning4j" % "deeplearning4j-nlp" % "1.0.0-M2.1"

libraryDependencies += "org.nd4j" % "nd4j-native-platform" % "1.0.0-M2.1"

