package cc.refectorie.Encoding

import java.util.Random

/**
 * Created by IntelliJ IDEA.
 * User: apassos
 * Date: 1/25/12
 * Time: 11:22 AM
 * To change this template use File | Settings | File Templates.
 */

class ParallelKMeans(data: Array[Array[(Int, Double)]], k: Int,  d: Int) {

  val means = Array.ofDim[Array[Double]](k)
  for (i <- 0 until k) means(i) = Array.ofDim[Double](d)
  val norms = Array.ofDim[Double](k)

  val rng = new java.util.Random()

  def sparseDot(d: Array[(Int,  Double)], m: Array[Double]) = {
    var dot = 0.0
    var j = 0
    while (j < d.length) {
      dot += m(d(j)._1)*d(j)._2
      j += 1
    }
    dot
  }
  // (x-y)^2 = x^2 - 2xy + y^2
  // note that x^2 is constant, so we can ignore it
  def dist(d: Array[(Int,  Double)], m: Int) = norms(m) - 2*sparseDot(d, means(m))


  def addTo(v: Array[Double], x: Array[(Int, Double)]) { x.foreach(e => v(e._1) += e._2) }

  def initialize() {
    for (i <- 0 until k) {
      addTo(means(i), data(rng.nextInt(data.length)))
      norms(i) = means(i).foldLeft(0.0)((sum, element) => sum + element*element)
    }
  }

  def eStep() = {
    data.par.map(d => {
      (0 until k).map(i => (i,  dist(d, i))).minBy(a => a._2)
    }).seq.toArray
  }

  def mStep(e: Array[(Int,Double)]) = {
    //e.zipWithIndex.groupBy(f =>  f._1._1).map(group => {
    e.zipWithIndex.par.groupBy(f => f._1._1).map(group => {
      val mean = group._1
      val examples = group._2
      val newMean = means(mean)
      var i = 0
      while (i < newMean.length) {newMean(i) = 0.0; i+=1}
      examples.foreach(e => addTo(newMean, data(e._2)))
      val error = examples.map(_._1._2).sum
      while (i < newMean.length) {
        newMean(i) /= examples.length
        i += 1
      }
      norms(mean) = newMean.foldLeft(0.0)((sum, e) => sum + e*e)
      error
    }).sum
  }

  def process(iterations: Int, tolerance: Double = 0.001) = {
    var err = 1000000000.0
    var oldError = Double.PositiveInfinity
    var i = 0
    while ((i < iterations) && (oldError - err > tolerance)) {
      oldError = err
      err = mStep(eStep())
      i += 1
    }
    err
  }

}

object ParallelKMeans {
  def main(args: Array[String]) {
    val rng = new java.util.Random()
    val data = Array.ofDim[Array[(Int, Double)]](200)
    for (i <- 0 until data.length) {
      data(i) = Array.ofDim[(Int,  Double)](20)
      for (j <- 0 until data(i).length) {
        data(i)(j) = (rng.nextInt(100), rng.nextGaussian())
      }
    }

    val km = new ParallelKMeans(data, 20, 100)
    km.initialize()
    km.process(30)
  }

  // Runs the parallel KMeans n times and returns the means of the best run
  def bestOfN(data: Array[Array[(Int,  Double)]], k: Int,  n: Int,  d: Int, iterations: Int) = {
    var bestKM: ParallelKMeans = null
    var bestErr = Double.PositiveInfinity
    for (i <- 0 until n) {
      val km = new ParallelKMeans(data, k, d)
      km.initialize()
      val newerr = km.process(iterations)
      if (newerr < bestErr) {
        bestKM = km
        bestErr = newerr
      }
    }
    bestKM.means
  }
}