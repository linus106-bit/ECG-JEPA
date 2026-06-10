import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import kotlin.math.exp

/**
 * Minimal ONNX Runtime inference example for a 1-channel, 500 Hz ECG classifier.
 * The classifier ONNX exported from a fine-tuned checkpoint expects 2.5-second
 * crops by default: input shape [batch, 1, 1250]. To match finetune.py eval,
 * this example creates 2.5-second crops with 1.25-second stride, averages crop
 * logits, then applies sigmoid.
 */
class EcgClassifierOnnx(modelBytes: ByteArray) : AutoCloseable {
  private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
  private val session: OrtSession = env.createSession(modelBytes, OrtSession.SessionOptions())

  /**
   * @param ecg 1-channel ECG sampled at 500 Hz. This example uses a fixed 10-second window.
   * @return six classifier probabilities in order: AFIB, 1AVB, 2AVB, SVTAC, PAC, PVC.
   */
  fun infer(ecg: FloatArray): FloatArray {
    val window = fitToWindow(ecg)
    val crops = makeCrops(window) // [numCrops=7, channels=1, samples=1250]

    OnnxTensor.createTensor(env, crops).use { tensor ->
      session.run(mapOf(INPUT_NAME to tensor)).use { result ->
        @Suppress("UNCHECKED_CAST")
        val cropLogits = result[LOGITS_OUTPUT_INDEX].value as Array<FloatArray> // [numCrops, 6]
        return sigmoid(meanLogits(cropLogits))
      }
    }
  }

  private fun fitToWindow(ecg: FloatArray): FloatArray {
    val window = FloatArray(INPUT_SAMPLES)
    val copyLength = minOf(ecg.size, INPUT_SAMPLES)
    System.arraycopy(ecg, 0, window, 0, copyLength)
    return window
  }

  private fun makeCrops(window: FloatArray): Array<Array<FloatArray>> {
    val numCrops = ((INPUT_SAMPLES - CROP_SAMPLES) / CROP_STRIDE_SAMPLES) + 1
    return Array(numCrops) { cropIndex ->
      val start = cropIndex * CROP_STRIDE_SAMPLES
      arrayOf(window.copyOfRange(start, start + CROP_SAMPLES))
    }
  }

  private fun meanLogits(cropLogits: Array<FloatArray>): FloatArray {
    val output = FloatArray(cropLogits[0].size)
    for (logits in cropLogits) {
      for (i in output.indices) {
        output[i] += logits[i]
      }
    }
    for (i in output.indices) {
      output[i] /= cropLogits.size.toFloat()
    }
    return output
  }

  private fun sigmoid(logits: FloatArray): FloatArray {
    return FloatArray(logits.size) { i ->
      (1.0 / (1.0 + exp(-logits[i].toDouble()))).toFloat()
    }
  }

  override fun close() {
    session.close()
  }

  companion object {
    private const val INPUT_NAME = "ecg"
    private const val LOGITS_OUTPUT_INDEX = 0
    private const val SAMPLE_RATE_HZ = 500
    private const val WINDOW_SECONDS = 10
    private const val INPUT_SAMPLES = SAMPLE_RATE_HZ * WINDOW_SECONDS
    private const val CROP_SAMPLES = 1250
    private const val CROP_STRIDE_SAMPLES = 625

    val LABELS = arrayOf("AFIB", "1AVB", "2AVB", "SVTAC", "PAC", "PVC")
  }
}
