package cc.refectorie.Encoding

import java.util.Random
import collection.mutable._
import java.io.File

/**
 * Created by IntelliJ IDEA.
 * User: apassos
 * Date: 1/26/12
 * Time: 9:43 AM
 * To change this template use File | Settings | File Templates.
 */

class SparseSparseCoding(dictionary: Array[Array[(Int,Double)]], lambda: Double, tolerance: Double) {
  /* The objective is to minimize (1/2)||Ax - b||^2 + lambda*|x|_1

   we can do this by coordinate descent. Taking the subgradient w.r.t. x_i and
   setting it to zero we get

   x_i = truncate(A^T(i) dot residual(x_i=0), lambda) / ||A^T(i)||^2

   where residual = b - A dot (x - x_i*e_i)
   (that is, b - A x' where x' is x with the i-th coordinate zeroed out)

   so if we have the current residual b - Ax, to get the residual we need we just
   add x_i A^T_i to it. After updating x_i, we subtract x_i A^T_i.

   Due to efficiency reasons, then, it's much better to work with A^T than with A.
   Hence this really optimizes (1/2)||A^Tx - b||^2 + lambda|x|_1

   We can stop when the change in ||residual|| after going over everything is
   smaller than a given tolerance.
   */

  def dot(a: Array[(Int, Double)], b: Array[(Int, Double)]): Double = {
    var dot = 0.0
    var i = 0
    var j = 0
    while((i < a.length) && (j < b.length)) {
      if (a(i)._1 == b(j)._1) {
        dot += a(i)._2 * b(j)._2
        i += 1
        j += 1
      } else if (a(i)._1 < b(j)._1) {
        i += 1
      } else {
        j += 1
      }
    }
    dot
  }
  def norm(v: Array[(Int, Double)]) = dot(v, v)
  val norms = dictionary.map(norm(_)).toArray
  def max(a: Double,  b: Double) = if (a > b) a else b
  def truncate(r: Double) = r// r.signum * max(0.0, r.abs - lambda)

  def addTo(v: Array[(Int,  Double)], a: Double,  b: Array[(Int,  Double)]) = {
    var i = 0
    var j = 0
    val n = ArrayBuffer[(Int, Double)]()
    while ((i < v.length) && (j < b.length)) {
      if (v(i)._1 == b(j)._1) {
        val nn = v(i)._2 + a*b(j)._2
        if (Math.abs(nn) > 0.000001) // taking care of sparsity
          n.append((v(i)._1,nn))
        j += 1
        i += 1
      } else if (v(i)._1 < b(j)._1) {
        n.append(v(i))
        i += 1
      } else {
        n.append((b(j)._1, a*b(j)._2))
        j += 1
      }
    }
    while (i < v.length) {n.append(v(i)); i+=1}
    while (j < b.length) {n.append((b(j)._1, a*b(j)._2)); j += 1}
    n.toArray
  }


  // In case anyone who comes after me wants to make this faster, obvious approaches are:
  //   1. make x or the means sparse
  //   2. use BLAS to optimize addTo and dot
  //   3. precompute the dot products between v and the dictionary and ignore all zeroes
  //   4. it should be possible to keep the residuals as a sparse list and precompute all dot products
  //      between means with each other and with x to then efficiently evaluate a new mean
  //      to be added to the residuals without ever touching all possible features
  def encode(v: Array[(Int, Double)], verbose: Boolean=false) = {
    val x = Array.ofDim[Double](dictionary.length)
    var residual = Array.ofDim[(Int, Double)](0)
    residual = addTo(residual, -1.0, v)
    var oldChanged = ArrayBuffer[Int]()
    val list = ArrayBuffer[(Int,Double)]()
    var d = 0
    while (d < dictionary.length) {
      val dor = scala.math.abs(dot(v, dictionary(d)))
      if (dor > 0.0)
        list.append((d,dor))
      d += 1
    }
    list.sortBy(a => -a._2).take(50).foreach(a => oldChanged.append(a._1))
    if (verbose) println("Starting to code. Considering "+oldChanged.length+" dictionary elements")
    var it = 0
    while (it < 6) {
      val changed = ArrayBuffer[Int]()
      var j = 0
      while (j < oldChanged.length) {
        val i = oldChanged(j)
        val oldi = x(i)
        residual = addTo(residual, x(i), dictionary(i))
        x(i) = truncate(dot(dictionary(i), residual))/norms(i)
        residual = addTo(residual, -x(i), dictionary(i))
        if (Math.abs(oldi - x(i)) > 0.0001) changed.append(i)
        j += 1
      }
      oldChanged = changed
      //error = norm(residual)
      it += 1
      if (verbose) println("    "+changed.length+" changes")
    }
    x
  }
}


object SparseSparseCoding {
  def oldMain(args: Array[String]) {
    val rng = new java.util.Random()
    val dict = Array.ofDim[Array[Double]](20)
    dict.zipWithIndex.foreach(ee => {
      val i = ee._2
      dict(i) = Array.ofDim[Double](30)
      dict(i).zipWithIndex.foreach(e => {
        val j = e._2
        dict(i)(j) = rng.nextGaussian()
      })
    })

    // this means that A has 30 rows and 20 columns, so b should have dimension 30 and x 20
    val b = new ArrayBuffer[(Int,  Double)]
    b.append((1, -5.0))
    b.append((10, 15.0))
    b.append((7, -1))
    b.append((23, 17.0))
    val coder = new SparseCoding(dict, 0.1, 0.0000001)
    val x = coder.encode(b.toArray)
  }

  def main(args: Array[String]) {
    val dict = Array.ofDim[Array[Double]](100)
    val mf: io.Source = io.Source.fromFile("/Users/apassos/data/means-2882645.txt")
    var id = 0
    for (line <- mf.getLines()) {
      dict(id) = Array.ofDim[Double](2882645)
      val l = line.split(" ")
      var i = 0
      while (i < l.length) {
        dict(id)(i) = l(i).toDouble
        i += 1
      }
      id += 1
    }
    println("Read the means")
    val feats = HashMap[String, Array[(Int, Double)]]()
    val ff = io.Source.fromFile("/Users/apassos/data/features.txt")
    for (line <- ff.getLines().take(1000)) {
      val l = line.split(" ")
      if (l.length > 1) {
        feats(l(0)) = l.drop(1).filter(s => s != "").map(s => {
          val e = s.split(":")
          (e(0).toInt, e(1).toDouble)
        }).toArray
      }
    }
    println("Read the features")

    val coder = new SparseCoding(dict, 0.1, 0.1)
    feats.values.foreach(v => coder.encode(v, true))
  }
}
