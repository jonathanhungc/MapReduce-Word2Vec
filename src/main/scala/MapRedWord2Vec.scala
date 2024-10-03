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
import org.slf4j.LoggerFactory
import com.typesafe.config.ConfigFactory

import scala.jdk.CollectionConverters._
import scala.collection.mutable.Map
import scala.collection.mutable.ListBuffer
import java.io.{DataInput, DataOutput}

/**
 * DoubleArrayWritable is used to represent the number of times a word appears (count), and its vector
 * representation (vector). This class is used for communication between the mapper and the reducer
 */
class DoubleArrayWritable() extends Writable {
  // these are vars, but they are not mutated for any object except when initialized and could be problematic do deserialize
  // if they where vals
  private var vector: ArrayWritable = new ArrayWritable(classOf[DoubleWritable])
  private var count: IntWritable = new IntWritable()

  // constructor to initialize fields
  def this(values: Array[Double], count: Int) = {
    this()
    this.vector = new ArrayWritable(classOf[DoubleWritable], values.map(new DoubleWritable(_)))
    this.count = new IntWritable(count)
  }

  // write method for serialization
  override def write(out: DataOutput): Unit = {
    vector.write(out)
    count.write(out)
  }

  // readFields method for deserialization
  override def readFields(in: DataInput): Unit = {
    vector.readFields(in)
    count.readFields(in)
  }

  // getters for data
  def getVector: Array[Writable] = vector.get()
  def getCount: Int = count.get()
}

/**
 * MapWord2Vec class contains the logic to perform map operations over input text data. It takes text data,
 * counts the number or times a word appears, creates a Word2Vec model with the input text, and sends to the
 * reducer each word with its tokenized version and its vector in the model as key: (word, token) and value: (vector)
 */
class MapperWord2Vec extends Mapper[LongWritable, Text, Text, DoubleArrayWritable] {

  private val log = LoggerFactory.getLogger(classOf[MapperWord2Vec])
  private val config = ConfigFactory.load()

  private val trainingSentencesList = new ListBuffer[String]()  // used to hold the training sentences
  private val wordCountMap = Map.empty[String, Int]   // map that holds the count of every word

  /**
   * map() takes a line of text, and for each word it creates its token and gets the count (number of times it appears)
   */
  override def map(key: LongWritable, value: Text, context: Mapper[LongWritable, Text, Text, DoubleArrayWritable]#Context): Unit = {
    val line: String = value.toString.toLowerCase
    val words = line.split("\\W+").filter(_.nonEmpty)

    // loop through each word in the line, and add count of word to map. Key is (word,token) and value is count
    words.foreach { word =>
      wordCountMap.updateWith(word) {
        case Some(count) => Some(count + 1) // Increment the count if the word is already in the map
        case None => Some(1) // Initialize with 1 if the word is not in the map
      }
    }
    trainingSentencesList += line // adding line sentence to set for later training
  }

  /**
   * cleanup() creates the Word2Vec model, trains it using the trainingSentencesList. Also, it takes each word,
   * gets its vector representation for the model, and writes to context with key: (word, token) and value: (count, vector)
   */
  override def cleanup(context: Mapper[LongWritable, Text, Text, DoubleArrayWritable]#Context): Unit = {

    val javaSentences: java.util.Collection[String] = trainingSentencesList.asJava

    val iter = new CollectionSentenceIterator(javaSentences)
    val t = new DefaultTokenizerFactory
    t.setTokenPreProcessor(new CommonPreprocessor)
    log.info("Building model....")
    val vec = new Word2Vec.Builder()
      .minWordFrequency(config.getInt("app.minWordFrequency"))
      .iterations(config.getInt("app.iterations"))
      .layerSize(config.getInt("app.layerSize"))   // number of dimensions for each vector
      .seed(config.getInt("app.seed"))
      .windowSize(config.getInt("app.windowSize"))
      .iterate(iter)
      .tokenizerFactory(t)
      .build

    log.info("Fitting Word2Vec model....")
    vec.fit()

    wordCountMap.foreach { case (word, count) =>

      log.info("Working on word: " + word)

      val vectorArr: Array[Double] = vec.getWordVector(word)    // get vector for word from model

      if (vectorArr == null) {
        log.warn(s"No vector found for word: ${word}")
      } else {

        val vector = new DoubleArrayWritable(vectorArr, count)  // create DoubleArrayWritable for a word
        log.info(s"Key: ${word} Vector: ${vector.toString} Count: ${vector.getCount}")

        // write to context, using key: (word, token) and value: (count, vector)
        context.write(new Text(word), vector)
        log.info(s"Successfully wrote word: $word and its vector to context")
      }
    }
  }
}

/**
 * ReduceWord2Vec class contains the logic to take keys (word, token) and their values (count, vector), sums up
 * the counts to get the total count for a word, and averages multiple vectors that a word could have.
 */
class ReducerWord2Vec extends Reducer[Text, DoubleArrayWritable, Text, Text] {

  private val registry = Encodings.newDefaultEncodingRegistry()
  private val enc = registry.getEncoding(EncodingType.CL100K_BASE)
  private val log = LoggerFactory.getLogger(classOf[ReducerWord2Vec])
  private val config = ConfigFactory.load()

  override def reduce(key: Text, values: java.lang.Iterable[DoubleArrayWritable], context: Reducer[Text, DoubleArrayWritable, Text, Text]#Context): Unit = {
    log.info(s"Reducing key: $key")
    val vectorDimensions = config.getInt("app.layerSize")   // used to determine how many dimensions are each vector
    val averageValues = Array.fill(vectorDimensions)(new DoubleWritable(0.0))

    // vars, but they are in method scope, used to count the number of vectors for a particular key and the number of
    // occurrences. It shouldn't cause problems since they are used only inside the reduce() method.
    var vectorCount = 0
    var wordCount = 0

    values.asScala.foreach { wordVector =>

      // get array from values, and cast to array of DoubleWritable
      log.info("Received DoubleArrayWritable: " + wordVector.toString)
      val doubleArray = wordVector.getVector.map(_.asInstanceOf[DoubleWritable].get())
      log.info(doubleArray.mkString("Array(", ", ", ")"))

      // add each value to average vector
      doubleArray.zipWithIndex.foreach { case (number, index) =>
        val updatedValue = averageValues(index).get() + number // add values from each vector
        averageValues(index).set(updatedValue) // update the value
      }
      vectorCount += 1
      wordCount += wordVector.getCount
    }

    log.info(s"Number of vectors: ${vectorCount}")

    // calculate average for each number in average vector
    averageValues.foreach { vectorNumber =>
      vectorNumber.set(vectorNumber.get() / vectorCount)
    }

    // write count of vectors and average vector to key
    val averageString = averageValues.map(_.get().toString).mkString(",") // make string of average vector
    log.info(wordCount.toString + "," + averageString)

    context.write(new Text(key + "," + enc.encode(key.toString).get(0).toString), new Text(wordCount.toString + "," + "[" + averageString + "]"))
  }
}

object Word2VecDriver {
  def main(args: Array[String]): Unit = {

    // Create a new Configuration and Job instance
    val conf = new Configuration()

    conf.set("mapreduce.output.textoutputformat.separator", ",")

    val job = Job.getInstance(conf, "Word2Vec Encoding Job")

    // Set the jar class (main class for the program)
    job.setJarByClass(Word2VecDriver.getClass)

    // Set Mapper and Reducer classes
    job.setMapperClass(classOf[MapperWord2Vec])
    job.setReducerClass(classOf[ReducerWord2Vec])

    // Set mapper key and value class
    job.setMapOutputKeyClass(classOf[Text])
    job.setMapOutputValueClass(classOf[DoubleArrayWritable])

    // Set output key and value types
    job.setOutputKeyClass(classOf[Text])
    job.setOutputValueClass(classOf[Text])

    // Set the input and output formats
    job.setInputFormatClass(classOf[TextInputFormat])
    job.setOutputFormatClass(classOf[TextOutputFormat[Text, Text]])

    // Specify input and output paths
    FileInputFormat.addInputPath(job, new Path(args(0)))
    FileOutputFormat.setOutputPath(job, new Path(args(1)))

//    FileInputFormat.addInputPath(job, new Path("src/main/resources/input"))
//    FileOutputFormat.setOutputPath(job, new Path("src/main/resources/output"))

    // Submit the job and wait for it to complete
    System.exit(if (job.waitForCompletion(true)) 0 else 1)

  }
}