import org.apache.hadoop.mapreduce.Reducer
import org.apache.hadoop.mapreduce.Mapper
import org.scalatest.flatspec.AnyFlatSpec
import org.apache.hadoop.io.{DataInputBuffer, DataOutputBuffer, DoubleWritable}
import org.apache.hadoop.io.{LongWritable, Text}
import org.mockito.Mockito.{atLeastOnce, mock, never, times, verify}
import org.mockito.ArgumentMatchers.any

import java.util.{List => JList}
import scala.jdk.CollectionConverters._
import com.knuddels.jtokkit.Encodings
import com.knuddels.jtokkit.api.EncodingType

/**
 * Testing serialization and deserialization of data from DoubleArrayWritable, which is used for communication
 * between the mapper and the reducer. These tests test initialization and serialization/deserialization process.
 */
class DoubleArrayWritableTest extends AnyFlatSpec {

  // testing initialization of the data
  "DoubleArrayWritable" should "initialize to store and give values correctly" in {
    val array = Array(0.1, 0.2, 0.3)
    val count = 3
    val writable = new DoubleArrayWritable(array, count)

    assert(writable.getCount == count)
    assert(writable.getVector.map(_.asInstanceOf[DoubleWritable].get()) sameElements array)
  }

  // testing serialization of the data
  "DoubleArrayWritable" should "serialize and deserialize correctly" in {
    val array = Array(1.0, 2.0, 3.0)
    val count = 3
    val writable = new DoubleArrayWritable(array, count)

    // serialize
    val outBuffer = new DataOutputBuffer()
    writable.write(outBuffer)

    // deserialize
    val inBuffer = new DataInputBuffer()
    inBuffer.reset(outBuffer.getData, outBuffer.getLength)

    val deserializedWritable = new DoubleArrayWritable()
    deserializedWritable.readFields(inBuffer)

    // assertions to check serialization/deserialization process
    assert(deserializedWritable.getCount == count)
    assert(deserializedWritable.getVector.map(_.asInstanceOf[DoubleWritable].get()) === array)
  }
}

/**
 * Testing for the mapper class (MapperWord2Vec). It initializes an instance of MapperWord2Vec, creates a mock
 * context for the mapper. Takes a test string, passes it through the map() function, and the test verifies
 * that the map is writing to the context with the format of (word, DoubleArrayWritable)
 */
class MapperWord2VecTest extends AnyFlatSpec {

  // checking the core functionality of the map() function
  "MapperWord2Vec" should "process a line of text and add it for training" in {
    val mapper = new MapperWord2Vec()
    val context = mock(classOf[Mapper[LongWritable, Text, Text, DoubleArrayWritable]#Context])

    val text = new Text("hello world this is a test")
    mapper.map(new LongWritable(1), text, context)

    // after mapping, we'll verify that cleanup will process sentences as expected
    mapper.cleanup(context)

    // verify if the cleanup function has trained the model and called context.write
    verify(context, atLeastOnce()).write(org.mockito.ArgumentMatchers.eq(new Text("hello")), any[DoubleArrayWritable])
    verify(context, atLeastOnce()).write(org.mockito.ArgumentMatchers.eq(new Text("world")), any[DoubleArrayWritable])
    verify(context, atLeastOnce()).write(org.mockito.ArgumentMatchers.eq(new Text("test")), any[DoubleArrayWritable])
  }

  // checking the functionality of repeated words
  "MapperWord2Vec" should "handle duplicate words correctly" in {
    val mapper = new MapperWord2Vec()
    val context = mock(classOf[Mapper[LongWritable, Text, Text, DoubleArrayWritable]#Context])

    val text = new Text("hello hello hello world world this is a test")
    mapper.map(new LongWritable(1), text, context)

    // call cleanup to write to context
    mapper.cleanup(context)

    // verify that everything was written just once,
    verify(context, times(1)).write(org.mockito.ArgumentMatchers.eq(new Text("hello")), any[DoubleArrayWritable])
    verify(context, times(1)).write(org.mockito.ArgumentMatchers.eq(new Text("world")), any[DoubleArrayWritable])
    verify(context, times(1)).write(org.mockito.ArgumentMatchers.eq(new Text("test")), any[DoubleArrayWritable])
    verify(context, never()).write(org.mockito.ArgumentMatchers.eq(new Text("nope")), any[DoubleArrayWritable]) // shouldn't write
  }
}

/**
 * Testing for the reducer class (ReducerWord2Vec). It initializes an instance of ReducerWord2Vec, creates a mock
 * context for the reducer. Uses a test string (word, a key), creates an iterable of two DoubleArrayWritable objects,
 * and calls the reduce() function. It verifies that the reduce() function is writing to context with
 * key: (word, token) and value: (count, average vector)
 */
class ReducerWord2VecTest extends AnyFlatSpec {
  private val registry = Encodings.newDefaultEncodingRegistry()
  private val enc = registry.getEncoding(EncodingType.CL100K_BASE)

  "ReducerWord2Vec" should "process a key (word), value (iterable of DoubleArrayWritable) pair and output the" +
    " key (word, token) and value (count, average vector)" in {
    val reducer = new ReducerWord2Vec()
    val context = mock(classOf[Reducer[Text, DoubleArrayWritable, Text, Text]#Context])

    val textString = "test"
    val text = new Text(textString)

    // creating input arrays. Each pair or numbers sums to 0.9 (0.4 + 0.5). It uses 10 numbers, considering
    // that each vector as 10 dimensions (this can be configured in the program, in the config file)
    val array1 = new DoubleArrayWritable(Array(0.5, 0.4, 0.4, 0.4, 0.5, 0.4, 0.5, 0.5, 0.5, 0.4), 3)
    val array2 = new DoubleArrayWritable(Array(0.4, 0.5, 0.5, 0.5, 0.4, 0.5, 0.4, 0.4, 0.4, 0.5), 2)
    val inputValues: JList[DoubleArrayWritable] = List(array1, array2).asJava   // create iterable for objects

    // call reduce() to write to context
    reducer.reduce(text, inputValues, context)

    // verify if the reducer gave as output the word, its token, its count, and the average vector for the word
    verify(context, atLeastOnce()).write(org.mockito.ArgumentMatchers.eq(new Text(textString + ","
      + enc.encode(textString).get(0).toString)),
      org.mockito.ArgumentMatchers.eq(new Text("5,[0.45,0.45,0.45,0.45,0.45,0.45,0.45,0.45,0.45,0.45]")))
  }
}