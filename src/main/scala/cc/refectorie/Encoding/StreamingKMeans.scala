package cc.refectorie.Encoding

import cc.factorie.la.{DenseTensor1, Tensor}

/**
 * Created by IntelliJ IDEA.
 * User: apassos
 * Date: 2/10/12
 * Time: 10:53 AM
 * To change this template use File | Settings | File Templates.
 */

object StreamingKMeans {

  /**
   * Does a streaming clustering algorithm. Each point is processed
   * in turn, and added to the best cluster with a probability proportional
   * to its distance.
   * @param data The points to be clustered
   * @param f If more than 1, less clusters are formed, if less than 1, more clusters
   * @return The cluster centers found
   */
  def process(data: Seq[Tensor], f: Double = 1.0) = {
    val centers = collection.mutable.ArrayBuffer[(Tensor,Double)]()
    val n0 = new DenseTensor1(data(0).length)
    n0 += data(0)
    centers.append((n0,1))
    val rng = new util.Random()
    var i = 1
    while (i < data.length) {
      val x = data(i)
      val (dist, bestI) = centers.zipWithIndex.par.map({ case ((t,s),idx) => (2.0 - 2*t.dot(x)/s,idx)}).maxBy(_._1)
      if (rng.nextDouble() < f*dist/centers.length) {
        val n = new DenseTensor1(x.length)
        n += x
        centers.append((n,1.0))
      } else {
        val (t,d) = centers(bestI)
        t += x
        centers(bestI) = (t,d+1)
      }
      i += 1
      if (i % 1000 == 0) {
        println("iter "+i+" with "+centers.length+" centers")
      }
    }

    centers.map({ case (t,d) => t /= d; t})
  }


}
