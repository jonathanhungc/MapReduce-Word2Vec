import org.apache.hadoop.fs.Path
import org.apache.hadoop.conf._
import org.apache.hadoop.io._
import org.apache.hadoop.mapreduce._
import org.apache.hadoop.mapreduce.Reducer
import org.apache.hadoop.mapreduce.Mapper


import org.apache.hadoop.mapreduce.lib.input.TextInputFormat
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat

import com.knuddels.jtokkit.Encodings
import com.knuddels.jtokkit.api.EncodingType
import org.deeplearning4j.models.word2vec.Word2Vec
import org.deeplearning4j.text.sentenceiterator.CollectionSentenceIterator
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory

import org.slf4j.Logger
import org.slf4j.LoggerFactory

import scala.jdk.CollectionConverters._
import scala.collection.mutable.Set
import scala.collection.mutable.ListBuffer

class MapWord2Vec extends Mapper[LongWritable, Text, Text, ArrayWritable] {
  val registry = Encodings.newDefaultEncodingRegistry()
  val enc = registry.getEncoding(EncodingType.CL100K_BASE)
  private val log = LoggerFactory.getLogger(classOf[MapWord2Vec])

  private val words_tokens_set = Set.empty[String]
  private val sentences_list = new ListBuffer[String]()

  override def map(key: LongWritable, value: Text, context: Mapper[LongWritable, Text, Text, ArrayWritable]#Context): Unit = {
    val line: String = value.toString.toLowerCase
    val words = line.split("\\W+")
    // log.info("Mappppppp start")
    sentences_list += line // adding line sentence to set for later training
    words.foreach(word =>
      words_tokens_set += (word + "," + enc.encode(word).get(0).toString)) // create keys (word,token) for later

    //    val encoded = line.split(" ").map(enc.encode)
    //    words.foreach()

    //    encoded.foreach { token =>
    //      context.write(new Text(token.get(0).toString), new IntWritable(1))
    //    }
    // log.info("Mappppppp end")

  }

  override def cleanup(context: Mapper[LongWritable, Text, Text, ArrayWritable]#Context): Unit = {
    // Convert the Scala collection to a Java collection
    val javaSentences: java.util.Collection[String] = sentences_list.asJava

    log.info("Cleannnnnnnn start")
    val iter = new CollectionSentenceIterator(javaSentences)
    val t = new DefaultTokenizerFactory

    t.setTokenPreProcessor(new CommonPreprocessor)
    log.info("Building model....")
    val vec = new Word2Vec.Builder()
      .minWordFrequency(1)
      .iterations(1)
      .layerSize(5)
      .seed(42)
      .windowSize(5)
      .iterate(iter)
      .tokenizerFactory(t)
      .build

    log.info("Fitting Word2Vec model....")
    vec.fit()


    //log.info("Set of words and tokens: " + words_tokens_set.mkString(", "))
    words_tokens_set.foreach { word =>
      try {
        log.info("Working on word: " + word)

        val vectorArr: Array[Double] = vec.getWordVector(word.split(",")(0))    // get vector for word

        if (vectorArr == null) {
          log.warn(s"No vector found for word: ${word.split(",")(0)}")
        } else {

          // log.info(vectorArr.mkString("Array(", ", ", ")"))   // print vector for debugging

          try {
            // create Array of DoubleWritable
            val doubleWritableArray: Array[Writable] = vectorArr.map(value => new DoubleWritable(value))
            log.info("Successfully created doubleWritableArray")

            // create ArrayWritable
            val vector = new ArrayWritable(classOf[DoubleWritable], doubleWritableArray)
            log.info("Successfully created ArrayWritable")
            log.info(s"Key: ${word} Vector: ${vector.toString}")
            // write to context
            context.write(new Text(word), vector)
            log.info(s"Successfully wrote word: $word and its vector to context")
          } catch {
            case e: Exception => log.error(s"Error in array or writing process for word: $word", e)
          }
        }
      } catch {
        case e: Exception => log.error(s"Error processing word: $word", e)
      }
      log.info("Cleannnnnnnn end")
    }

    // function to use array of doubles and write an ArrayWritable of DoubleWritable
    def getVector(vector: Array[Double]): ArrayWritable = {
    val doubleWritableArray: Array[Writable] = vector.map(value => new DoubleWritable(value))
    new ArrayWritable(classOf[DoubleWritable], doubleWritableArray)
    }
  }
}

class ReduceWord2Vec extends Reducer[Text, ArrayWritable, Text, Text] {

  private val log = LoggerFactory.getLogger(classOf[ReduceWord2Vec])

  override def reduce(key: Text, values: java.lang.Iterable[ArrayWritable], context: Reducer[Text, ArrayWritable, Text, Text]#Context): Unit = {
    log.info("Redddddddd start")
    log.info(s"Reducing key: $key")
    val arrayLength = 100
    val averageValues = Array.fill(arrayLength)(new DoubleWritable(0.0))
    var count = 0

    values.asScala.foreach { arrayWritable =>
      // get array from values, and cast to array of DoubleWritable
      val doubleArray = arrayWritable.get().map(_.asInstanceOf[DoubleWritable].get())

      // add each value to average vector
      doubleArray.zipWithIndex.foreach { case (number, index) =>
        val updatedValue = averageValues(index).get() + number // add values from each vector
        averageValues(index).set(updatedValue) // update the value
      }
      count += 1
    }

    // calculate average for each number in average vector
    averageValues.foreach { vectorNumber =>
      vectorNumber.set(vectorNumber.get() / count)
    }

    // write count of vectors and average vector to key
    val averageString = averageValues.map(_.get().toString).mkString(" ")   // make string of average vector
    context.write(key, new Text(count.toString + "," + averageString))

    //      val doubleWritableArray: Array[Writable] = averageValues.map(value => value)
    //      val resultArrayWritable = new ArrayWritable(classOf[DoubleWritable], doubleWritableArray)
    //      context.write(new Text("," + key + "," + count.toString), resultArrayWritable)

    log.info("Redddddddd start")
  }
}

object Word2VecDriver {
  def main(args: Array[String]): Unit = {

    // Create a new Configuration and Job instance
    val conf = new Configuration()

    conf.set("mapreduce.output.textoutputformat.separator", ",")
//    conf.set("fs.defaultFS", "local")
//    conf.set("mapreduce.job.maps", "1")
//    conf.set("mapreduce.job.reduces", "1")

    val job = Job.getInstance(conf, "Word2Vec Encoding Job")

    // Set the jar class (main class for the program)
    job.setJarByClass(Word2VecDriver.getClass)

    // Set Mapper and Reducer classes
    job.setMapperClass(classOf[MapWord2Vec])
    job.setReducerClass(classOf[ReduceWord2Vec])

    // Set mapper key and value class
    job.setMapOutputKeyClass(classOf[Text])
    job.setMapOutputValueClass(classOf[ArrayWritable])

    // Set output key and value types
    job.setOutputKeyClass(classOf[Text])
    job.setOutputValueClass(classOf[Text])

    // Set the input and output formats
    job.setInputFormatClass(classOf[TextInputFormat])
    job.setOutputFormatClass(classOf[TextOutputFormat[Text, Text]])

    // Specify input and output paths
    FileInputFormat.addInputPath(job, new Path("src/main/resources/input"))
    FileOutputFormat.setOutputPath(job, new Path("src/main/resources/output"))

    // Submit the job and wait for it to complete
    System.exit(if (job.waitForCompletion(true)) 0 else 1)

  }
}