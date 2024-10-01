package vector

import org.apache.hadoop.io.Writable
import java.io.{DataInput, DataOutput, IOException}
import scala.collection.mutable.ArrayBuffer
import org.apache.hadoop.io.ArrayWritable
import org.apache.hadoop.io.DoubleWritable

class Vector(var values: Array[Double], var count: Int) extends Writable {

  def this(originalValues: Array[Double]) = {
    this(new Array[Double](originalValues.length), 1)
    Array.copy(originalValues, 0, this.values, 0, originalValues.length)
  }

  def this(arraySize: Int) = {
    this(new Array[Double](arraySize), 1) // Initialize array with specified size
  }

  def add_vectors(other_vector: Vector): Unit = {
    if (this.values.length != other_vector.values.length) {
      throw new IllegalArgumentException("Vectors must be of the same size")
    }

    this.values.zipWithIndex.foreach { case (value, index) =>
      this.values(index) = value + other_vector.values(index) // Update this.values in place
    }

    count += 1
  }

  // Display information about the vector
  def showInfo(): Unit = {
    println(s"Vector: ${values.mkString(", ")}, Count: $count")
  }

  // Implementing Writable methods
  override def write(out: DataOutput): Unit = {
    out.writeInt(count)
    out.writeInt(values.length)
    values.foreach(out.writeDouble)
  }

  override def readFields(in: DataInput): Unit = {
    count = in.readInt()
    val length = in.readInt()
    values = new Array[Double](length)
    for (i <- values.indices) {
      values(i) = in.readDouble()
    }
  }
}
