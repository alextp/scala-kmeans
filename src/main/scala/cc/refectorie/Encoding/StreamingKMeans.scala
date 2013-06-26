package cc.refectorie.Encoding

import java.io.{FileWriter, BufferedWriter}

/**
 * Created by IntelliJ IDEA.
 * User: apassos
 * Date: 2/10/12
 * Time: 10:53 AM
 * To change this template use File | Settings | File Templates.
 */

object StreamingKMeans {
  val beta = 10.0

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

  def process(data: Array[Array[(Int, Double)]], k: Int,  d: Int, kappa: Int = 0, weights: Array[Double] = null) = {
    val centers = collection.mutable.ArrayBuffer[Array[(Int,  Double)]]()
    val kp = if (kappa > 0) kappa else 10*k*Math.log(data.length)
    val ws = if (weights != null) weights else Array.ofDim[Int](data.length)
    var f = 1.0/(k*(1.0+ Math.log(data.length)))
    val rng = new util.Random()
    var i = 1
    centers.append(data(0))
    while (i < data.length) {
      //if (centers.length < kp) {
      val x = data(i)
      val dist = 2.0 - 2*centers.par.map(c => dot(x, c)).max
      if (rng.nextDouble() < dist/(centers.length)) {
        centers.append(x)
      }
      i += 1
      if (i % 1000 == 0) {
        println("iter "+i+" with "+centers.length+" centers")
      }
    }

    centers
  }


}
