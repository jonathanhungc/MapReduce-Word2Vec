# Homework 1: MapReduce
### Name: Jonathan Hung

#### UIC email: jhung9@uic.edu

This is a program to run Hadoop MapReduce over a corpus of text. The data is partitioned in different books that are
used as input for the program. You can see two directories: src/main/resources/input and src/main/resources/input-large.
The directory input holds little data to run the program, while the data in input-large is more extensive (but not as
extensive as the data used for running the program on Amazon EMR). So, the important files and directories are:

- **src/main/resources**: Holds all the input data for the program, with directories with small data (/input) and large data
(/input-large). It also includes the /output directory where the results of the execution of the program are. Also,
it includes the application.conf file, which is the configuration file, mostly used for the implementation of Word2Vec.
- **src/main/scala/MapRedWord2Vec.scala**: This is the file that contains all the classes for the program (MapperWord2Vec, 
ReducerWord2Vec, Word2VecDriver, as well as DoubleArrayWritable, which is used for serialization
/ deserialization for objects passed between mapper and reducer).
- **src/test/scala/Test.scala**: This file has all the tests for the classes in MapRedWord2Vec.scala. Tests the
serialization/deserialization of DoubleArrayWritable, and that the mapper and reducer functions are writing
to context correctly. The test file has mock contexts to test these functionalities.
- **build.sbt**: This file has all the dependencies of the program, including Apache Hadoop, DeepLearning4j, etc.
- **Input**: A directory with .txt files with sentences
- **Output**: A directory with files, after processing the text, where each entry in a file is comma separated as:
(word,token,count,vector)
### To run the program execute the following commands:
```
sbt clean
sbt compile
sbt "run src/main/resources/input src/main/resources/output"
```

### The video of the deployment on Amazon EMR (Elastic Map Reduce) is found [here](https://youtu.be/qI8PZPiBnFM)