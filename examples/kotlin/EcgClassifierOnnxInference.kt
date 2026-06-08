import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession

/**
 * Minimal ONNX Runtime inference example for a 1-channel, 500 Hz ECG classifier.
 * The ONNX model is expected to have input shape [batch, 1, 5000] and to return
 * outputs named "logits" and "probabilities" from export_ecg_encoder_onnx.py.
 */
class EcgClassifierOnnx(modelBytes: ByteArray) : AutoCloseable {
  private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
  private val session: OrtSession = env.createSession(modelBytes, OrtSession.SessionOptions())

  /**
   * @param ecg 1-channel ECG sampled at 500 Hz. This example uses a fixed 10-second window.
   * @return six classifier probabilities in order: AFIB, 1AVB, 2AVB, SVTAC, PAC, PVC.
   */
  fun infer(ecg: FloatArray): FloatArray {
    val input = arrayOf(arrayOf(fitToWindow(ecg))) // [batch=1, channels=1, samples=5000]

    OnnxTensor.createTensor(env, input).use { tensor ->
      session.run(mapOf(INPUT_NAME to tensor)).use { result ->
        @Suppress("UNCHECKED_CAST")
        val probabilities = result[PROBABILITIES_OUTPUT_INDEX].value as Array<FloatArray> // [1, 6]
        return probabilities[0]
      }
    }
  }

  private fun fitToWindow(ecg: FloatArray): FloatArray {
    val window = FloatArray(INPUT_SAMPLES)
    val copyLength = minOf(ecg.size, INPUT_SAMPLES)
    System.arraycopy(ecg, 0, window, 0, copyLength)
    return window
  }

  override fun close() {
    session.close()
  }

  companion object {
    private const val INPUT_NAME = "ecg"
    private const val PROBABILITIES_OUTPUT_INDEX = 1
    private const val SAMPLE_RATE_HZ = 500
    private const val WINDOW_SECONDS = 10
    private const val INPUT_SAMPLES = SAMPLE_RATE_HZ * WINDOW_SECONDS

    val LABELS = arrayOf("AFIB", "1AVB", "2AVB", "SVTAC", "PAC", "PVC")
  }
}
