/*
 * Copyright 2018 Zihua Zeng (edvard_hua@live.com), Lang Feng (tearjeaker@hotmail.com)
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.edvard.poseestimation

import android.app.Activity
import android.util.Log

import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc

import java.io.IOException

/**
 * Pose Estimator
 */
class ImageClassifierFloatInception private constructor(
    activity: Activity,
    imageSizeX: Int,
    imageSizeY: Int,
    private val outputW: Int,
    private val outputH: Int,
    modelPath: String,
    numBytesPerChannel: Int = 4 // a 32bit float value requires 4 bytes
  ) : ImageClassifier(activity, imageSizeX, imageSizeY, modelPath, numBytesPerChannel) {

  /**
   * An array to hold inference results, to be feed into Tensorflow Lite as outputs.
   * This isn't part of the super class, because we need a primitive array here.
   */
  private val heatMapArray: Array<Array<Array<FloatArray>>> =
    Array(1) { Array(outputW) { Array(outputH) { FloatArray(14) } } }

  private var mMat: Mat? = null

  override fun addPixelValue(pixelValue: Int) {
    //bgr
    imgData!!.putFloat((pixelValue and 0xFF).toFloat())
    imgData!!.putFloat((pixelValue shr 8 and 0xFF).toFloat())
    imgData!!.putFloat((pixelValue shr 16 and 0xFF).toFloat())
  }

  override fun getProbability(labelIndex: Int): Float {
    //    return heatMapArray[0][labelIndex];
    return 0f
  }

  override fun setProbability(
    labelIndex: Int,
    value: Number
  ) {
    //    heatMapArray[0][labelIndex] = value.floatValue();
  }

  override fun getNormalizedProbability(labelIndex: Int): Float {
    return getProbability(labelIndex)
  }

  override fun runInference() {
    tflite?.run(imgData!!, heatMapArray)

    if (mPrintPointArray == null)
      mPrintPointArray = Array(2) { FloatArray(14) }

    if (!CameraActivity.isOpenCVInit)
      return

    // Gaussian Filter 5*5
    if (mMat == null)
      mMat = Mat(outputW, outputH, CvType.CV_32F)

    val tempArray = FloatArray(outputW * outputH)
    val outTempArray = FloatArray(outputW * outputH)

    var rwrs_x: Float = 0.0f
    var rwrs_y: Float = 0.0f
    var lwrs_x: Float = 0.0f
    var lwrs_y: Float = 0.0f
    var relb_x: Float = 0.0f
    var relb_y: Float = 0.0f
    var lelb_x: Float = 0.0f
    var lelb_y: Float = 0.0f
    var rshl_x: Float = 0.0f
    var rshl_y: Float = 0.0f
    var lshl_x: Float = 0.0f
    var lshl_y: Float = 0.0f

    for (i in 0..13) {
      var index = 0
      for (x in 0 until outputW) {
        for (y in 0 until outputH) {
          tempArray[index] = heatMapArray[0][y][x][i]
          index++
        }
      }

      mMat!!.put(0, 0, tempArray)
      Imgproc.GaussianBlur(mMat!!, mMat!!, Size(5.0, 5.0), 0.0, 0.0)
      mMat!!.get(0, 0, outTempArray)

      var maxX = 0f
      var maxY = 0f
      var max = 0f

      // Find keypoint coordinate through maximum values
      for (x in 0 until outputW) {
        for (y in 0 until outputH) {
          val center = get(x, y, outTempArray)
          if (center > max) {
            max = center
            maxX = x.toFloat()
            maxY = y.toFloat()
          }
        }
      }

      if (max == 0f) {
        mPrintPointArray = Array(2) { FloatArray(14) }
        return
      }

      mPrintPointArray!![0][i] = maxX
      mPrintPointArray!![1][i] = maxY
//      Log.i("TestOutPut", "pic[$i] ($maxX,$maxY) $max")


      when (i) {
        0 -> Log.i("Inference", " HEAD ($maxX, $maxY)")

        1 -> {
         Log.i("Inference", " NECK ($maxX, $maxY)")
//          neck_x = maxX
//          neck_y = maxY
        }

        2 -> {
          Log.i("Inference", " LSHL ($maxX, $maxY)")
          lshl_x = maxX
          lshl_y = maxY
        }
        5 -> {
          Log.i("Inference", " RSHL ($maxX, $maxY)")
          rshl_x = maxX
         rshl_y = maxY
        }

        3 -> {
          Log.i("Inference", " LELB ($maxX, $maxY)")
          lelb_x = maxX
         lelb_y = maxY
        }
        6 -> {
          Log.i("Inference", " RELB ($maxX, $maxY)")
          relb_x = maxX
         relb_y = maxY
        }

        4 -> {
          Log.i("Inference", " LWRS ($maxX, $maxY)")
         lwrs_x = maxX
          lwrs_y = maxY
        }
        7 -> {
         Log.i("Inference", " RWRS ($maxX, $maxY)")
          rwrs_x = maxX
          rwrs_y = maxY
        }

        8 -> Log.i("Inference", " LHIP ($maxX, $maxY)")
       11 -> Log.i("Inference", " RHIP ($maxX, $maxY)")

        9 -> Log.i("Inference", " LKNE ($maxX, $maxY)")
        12 -> Log.i("Inference", " RKNE ($maxX, $maxY)")

        10 -> Log.i("Inference", " LANK ($maxX, $maxY)")
       13 -> Log.i("Inference", " RANK ($maxX, $maxY)")
      }
    }
    // right hand up
    if (rwrs_y > relb_y && relb_y > rshl_y) {
      Log.e("shivam", "hands up")

    }
  }

  private operator fun get(
    x: Int,
    y: Int,
    arr: FloatArray
  ): Float {
    return if (x < 0 || y < 0 || x >= outputW || y >= outputH) -1f else arr[x * outputW + y]
  }

  companion object {

    /**
     * Create ImageClassifierFloatInception instance
     *
     * @param imageSizeX Get the image size along the x axis.
     * @param imageSizeY Get the image size along the y axis.
     * @param outputW The output width of model
     * @param outputH The output height of model
     * @param modelPath Get the name of the model file stored in Assets.
     * @param numBytesPerChannel Get the number of bytes that is used to store a single
     * color channel value.
     */
    fun create(
      activity: Activity,
      imageSizeX: Int = 192,
      imageSizeY: Int = 192,
      outputW: Int = 96,
      outputH: Int = 96,
      modelPath: String = "model.tflite",
      numBytesPerChannel: Int = 4
    ): ImageClassifierFloatInception =
      ImageClassifierFloatInception(
          activity,
          imageSizeX,
          imageSizeY,
          outputW,
          outputH,
          modelPath,
          numBytesPerChannel)
  }
}
