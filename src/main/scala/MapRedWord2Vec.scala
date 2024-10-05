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
 * representation (vector). This class is used for communication between the mapper and the reducer.
 * It extends Writable for serialization.
 */
class DoubleArrayWritable() extends Writable {
  // these are vars, but they are not mutated for any object except when initialized and could be problematic do deserialize
  // if they where vals
  private var vector: ArrayWritable = new ArrayWritable(classOf[DoubleWritable])
  private val count: IntWritable = new IntWritable(-1)

  // constructor to initialize fields
  def this(values: Array[Double], count: Int) = {
    this()
    this.vector = new ArrayWritable(classOf[DoubleWritable], values.map(new DoubleWritable(_)))
    this.count.set(count)
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
 * The Mapper takes care of doing the word counting for each word given from the input.
 */
class MapperWord2Vec extends Mapper[LongWritable, Text, Text, DoubleArrayWritable] {

  private val registry = Encodings.newDefaultEncodingRegistry()
  private val enc = registry.getEncoding(EncodingType.CL100K_BASE)
  private val log = LoggerFactory.getLogger(classOf[MapperWord2Vec])  // logger
  private val config = ConfigFactory.load() // config file

  // this ListBuffer is mutable, and is used to store all the sentences for the training. Since the training for
  // my model happens in cleanup(), I needed to accumulate the training sentences that are processed by all the map()
  // calls, so I can feed them all to the Word2Vec model. I thought about training the model on the go, using the text
  // from each map() call and pass those sentences directly for training, but I wasn't allowed to do that using
  // DeepLearning4j Word2Vec. I could have created a Word2Vec model for each map() call and pass the text, but
  // (1) it would require too much time to create all those models, and (2) each model would use fewer data for training,
  // which would give less accurate results for the vector embeddings of words. It's not the most efficient method,
  // since it could lead to race conditions and some sentences may not be added to the model. I would like to discuss
  // later how this could be improved or changed.
  private val trainingSentencesList = new ListBuffer[String]()

  // this is a mutable map. I use it to hold all the words that come from the sentences that the mapper takes. I need
  // this map, so that I can store all the different words that come from input and store their counts. I know this
  // is not the most optimal solution, since it can ultimately cause race conditions. I hope to check in the future
  // with the professor about how I could improve or change this.
  private val wordCountMap = Map.empty[String, Int]

  /**
   * map() takes lines of text, parses them by taking and separating the words, and for each word it counts the
   * number of times it appears and stores it. It doesn't write anything to context, that happens in cleanup(). Also,
   * it adds the lines of text to trainingSentencesList, which is used later for training.
   * @param key   Number of the line in the text
   * @param value   A line of text
   * @param context   The context to write to
   */
  override def map(key: LongWritable, value: Text, context: Mapper[LongWritable, Text, Text, DoubleArrayWritable]#Context): Unit = {
    try {
      val line: String = value.toString.toLowerCase
      val words = line.split("\\W+").filter(word => word.nonEmpty && word.forall(_.isLetter))

      log.info(s"map() - Adding line: ${line}")
      trainingSentencesList += line   // add line of text for later training

      words.foreach { word =>
        wordCountMap.updateWith(word) {
          case Some(count) => Some(count + 1) // increment the count if the word is already in the map
          case None => Some(1) // initialize with 1 if the word is not in the map
        }
      }
    } catch {
      case e: Exception =>
        log.error("map() - Error during map execution", e)
        throw e  // rethrow error
    }
  }

  /**
   * cleanup() creates the Word2Vec model, trains it using the trainingSentencesList. Also, it takes each word, creates
   * its token representation, gets its vector representation for the model, and writes to context with
   * key: (word, token) and value: (count, vector). This function generates the Word2Vec model, and writes to context
   * the word with its number of occurrences and  its vector representation in the model. I thought about getting the
   * token each word (using JTokkit) using the reducer and not here, since in the reducing stage a word would be
   * tokenized only once, while here would happen multiple times (since there are multiple mappers that would deal with
   * repeated words) but I decided to do it here since in MapReduce implementations usually run more mappers than
   * reducers (num mappers > num reducers). It would be a balance between more mappers performing more operations
   * of tokenization, or fewer reducers performing fewer operations of tokenization.
   * @param context The context to write to with key: (word, token) and value: (count, vector)
   */
  override def cleanup(context: Mapper[LongWritable, Text, Text, DoubleArrayWritable]#Context): Unit = {

    val javaSentences: java.util.Collection[String] = trainingSentencesList.asJava  // get iterator for training sentences

    //log.info(s"cleanup() - The training sentences: ${trainingSentencesList}")

    // setting up everything for the Word2Vec model
    val iter = new CollectionSentenceIterator(javaSentences)
    val t = new DefaultTokenizerFactory
    t.setTokenPreProcessor(new CommonPreprocessor)
    log.info("cleanup() - Building model....")
    val vec = new Word2Vec.Builder()
      .minWordFrequency(config.getInt("app.minWordFrequency"))
      .iterations(config.getInt("app.iterations"))
      .layerSize(config.getInt("app.layerSize"))   // number of dimensions for each vector
      .seed(config.getInt("app.seed"))
      .windowSize(config.getInt("app.windowSize"))
      .iterate(iter)
      .tokenizerFactory(t)
      .build

    log.info("cleanup() - Fitting Word2Vec model....")
    vec.fit()

    wordCountMap.foreach { case (word, count) =>  // for each word in the map, which holds all the words

      log.info("cleanup() - Getting vector for word: " + word)

      val vectorArr: Array[Double] = vec.getWordVector(word)    // get vector for word from model

      if (vectorArr == null) {
        log.warn(s"cleanup() - No vector found for word: ${word}")
      } else {

        val vector = new DoubleArrayWritable(vectorArr, count)  // create DoubleArrayWritable for a word
        //log.info(s"cleanup() - Key: ${word} Vector: ${vector.getVector.mkString("Array(", ", ", ")")} Count: ${count}")

        // write to context, using key: (word, token) and value: (count, vector)
        context.write(new Text(word + "," + enc.encode(word).get(0).toString), vector)
        log.info(s"cleanup() - Successfully wrote: $word and its vector to context")
      }
    }
  }
}

/**
 * ReduceWord2Vec class contains the logic to take keys (word, token) and their values (count, vector), sums up
 * the counts to get the total count for a word, and averages multiple vectors that a word could have.
 */
class ReducerWord2Vec extends Reducer[Text, DoubleArrayWritable, Text, Text] {

  private val log = LoggerFactory.getLogger(classOf[ReducerWord2Vec])
  private val config = ConfigFactory.load()

  /**
   * reduce() takes a key, and iterable of DoubleArrayWritable objects, and sums the counts of each object to get the
   * total count (number of occurrences) for each word, and takes all the vectors from the iterable to average them.
   * @param key   A word, with its token representation
   * @param values    An iterable of DoubleArrayWritable objects
   * @param context   A context to write to with key: (word, token) and value: (total count, average vector)
   */
  override def reduce(key: Text, values: java.lang.Iterable[DoubleArrayWritable], context: Reducer[Text, DoubleArrayWritable, Text, Text]#Context): Unit = {
    log.info(s"reduce() - Reducing key: $key")
    val vectorDimensions = config.getInt("app.layerSize")   // used to determine how many dimensions are each vector
    val averageValues = Array.fill(vectorDimensions)(new DoubleWritable(0.0))

    // vars, but they are in method scope, used to count the number of vectors for a particular key and the number of
    // occurrences. It shouldn't cause problems since they are used only inside the reduce() method.
    var vectorCount = 0   // to store the number of vectors for a particular word
    var wordCount = 0   // to store the number of occurrences for a particular word

    values.asScala.foreach { wordVector =>

      try {
        // get array from values, and cast to array of DoubleWritable
        log.info(s"reduce() - Received DoubleArrayWritable with vector: ${wordVector.toString}")
        val doubleArray = wordVector.getVector.map(_.asInstanceOf[DoubleWritable].get())    // convert to array
        log.info(s"reduce() - Key: ${key} Vector: ${wordVector.getVector.mkString("Array(", ", ", ")")}")

        // add each value to average vector
        doubleArray.zipWithIndex.foreach { case (number, index) =>
          val updatedValue = averageValues(index).get() + number // add values from each vector
          averageValues(index).set(updatedValue) // update the value
        }

        vectorCount += 1    // update number of vectors
        wordCount += wordVector.getCount    // update number of count
      } catch {
        case e: Exception =>
          log.error("reduce() - Error during reduce execution", e)
          throw e  // rethrow error
      }
    }

    log.info(s"reduce() - Key: ${key} Number of vectors: ${vectorCount} Word count: ${wordCount}")

    // calculate average for each number in average vector
    averageValues.foreach(vectorNumber => vectorNumber.set(vectorNumber.get() / vectorCount))

    // write count of vectors and average vector to key (the word and its token)
    val averageString = averageValues.map(_.get().toString).mkString(",") // make string of average vector

    log.info(s"reduce() - Writing: ${key} Count: ${wordCount} Vector: [" + averageString + "]")
    context.write(key, new Text(wordCount.toString + "," + "[" + averageString + "]"))
    log.info(s"reduce() - Successfully wrote ${key} to context")

  }
}

/**
 * This is the driver for the program. It sets the configuration for the MapReduce job. Input and output
 * paths are given as command line arguments
 */
object Word2VecDriver {
  def main(args: Array[String]): Unit = {

    // Create a new Configuration and Job instance
    val conf = new Configuration()

    conf.set("mapreduce.output.textoutputformat.separator", ",")

    val job = Job.getInstance(conf, "Word2Vec Encoding Job")

    // set the jar class (main class for the program)
    job.setJarByClass(Word2VecDriver.getClass)

    // set Mapper and Reducer classes
    job.setMapperClass(classOf[MapperWord2Vec])
    job.setReducerClass(classOf[ReducerWord2Vec])

    // Sset mapper key and value class
    job.setMapOutputKeyClass(classOf[Text])
    job.setMapOutputValueClass(classOf[DoubleArrayWritable])

    // set output key and value types
    job.setOutputKeyClass(classOf[Text])
    job.setOutputValueClass(classOf[Text])

    // set the input and output formats
    job.setInputFormatClass(classOf[TextInputFormat])
    job.setOutputFormatClass(classOf[TextOutputFormat[Text, Text]])

    // specify input and output paths
    FileInputFormat.addInputPath(job, new Path(args(0)))
    FileOutputFormat.setOutputPath(job, new Path(args(1)))

//    FileInputFormat.addInputPath(job, new Path("src/main/resources/input"))
//    FileOutputFormat.setOutputPath(job, new Path("src/main/resources/output"))

    // submit the job and wait for it to complete
    System.exit(if (job.waitForCompletion(true)) 0 else 1)

  }
}