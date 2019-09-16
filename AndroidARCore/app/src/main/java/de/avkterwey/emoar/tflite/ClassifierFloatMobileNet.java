/* Created by Berenice Terwey, September 2019
==============================================================================*/

package de.avkterwey.emoar.tflite;

import android.app.Activity;

import java.io.IOException;

/** This TensorFlowLite classifier works with the float MobileNet model. */
public class ClassifierFloatMobileNet extends Classifier {

  /**
   * An array to hold inference results, to be feed into Tensorflow Lite as outputs. This isn't part
   * of the super class, because we need a primitive array here.
   */
  private float[][] labelProbArray = null;

  /**
   * Initializes a {@code ClassifierFloatMobileNet}.
   *
   * @param activity
   */
  public ClassifierFloatMobileNet(Activity activity, Device device, int numThreads)
          throws IOException {
    super(activity, device, numThreads);
    labelProbArray = new float[1][getNumLabels()];
  }

  @Override
  public int getImageSizeX() {
    return 48; // 192
  }

  @Override
  public int getImageSizeY() {
    return 48;
  }

  @Override
  protected String getModelPath() {
    return "weights-best.tflite";
  }

  @Override
  protected String getLabelPath() {
    return "mobilenet_v2_fer.txt";
  }
  // 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral

  @Override
  protected int getNumBytesPerChannel() {
    return Float.SIZE / Byte.SIZE;
  } // 4


  @Override
  protected void addPixelValue(int pixelValue) {
    //Log.d("### ClassFloatMobile", "pixelValue= " + pixelValue + "; put " + ((pixelValue >> 16) & 0xFF));
    //imgData.putFloat((((pixelValue >> 16) & 0xFF)/225f - 0.485f) / 0.229f);
    //imgData.putFloat((((pixelValue >> 8) & 0xFF)/225f - 0.456f) / 0.224f);
    //imgData.putFloat(((pixelValue & 0xFF)/225f - 0.406f) / 0.225f);

    // hack: only use 1 channel , because the model was trained with grayscale images
    // TODO Rather convert the camera image to grayscale.
    imgData.putFloat(((pixelValue & 0xFF)/225f - 0.485f) / 0.229f);

  }

  @Override
  protected float getProbability(int labelIndex) {
    return labelProbArray[0][labelIndex];
  }

  @Override
  protected void setProbability(int labelIndex, Number value) {
    labelProbArray[0][labelIndex] = value.floatValue();
  }

  @Override
  protected float getNormalizedProbability(int labelIndex) {
    return labelProbArray[0][labelIndex];
  }

  @Override
  protected void runInference() {
    tflite.run(imgData, labelProbArray);
  }
}

