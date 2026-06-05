import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession

/**
 * Minimal ONNX Runtime inference example for a 1-channel, 500 Hz ECG encoder.
 * The ONNX model is expected to have input shape [batch, 1, 5000].
 *
 * Android dependency example:
 *   implementation("com.microsoft.onnxruntime:onnxruntime-android:<version>")
 *
 * Asset loading example:
 *   val modelBytes = context.assets.open("ecg_encoder.onnx").use { it.readBytes() }
 *   val encoder = EcgEncoderOnnx(modelBytes)
 *   val pooledEmbedding: FloatArray = encoder.infer(oneLeadEcg500Hz)
 */
class EcgEncoderOnnx(modelBytes: ByteArray) : AutoCloseable {
  private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
  private val session: OrtSession = env.createSession(modelBytes, OrtSession.SessionOptions())

  /**
   * @param ecg 1-channel ECG sampled at 500 Hz. This example uses a fixed 10-second window.
   * @return mean-pooled encoder embedding, shape: [embeddingDim].
   */
  fun infer(ecg: FloatArray): FloatArray {
    val input = arrayOf(arrayOf(fitToWindow(ecg))) // [batch=1, channels=1, samples=5000]

    OnnxTensor.createTensor(env, input).use { tensor ->
      session.run(mapOf(INPUT_NAME to tensor)).use { result ->
        @Suppress("UNCHECKED_CAST")
        val tokens = result[0].value as Array<Array<FloatArray>> // [1, numTokens, embeddingDim]
        return meanPool(tokens[0])
      }
    }
  }

  private fun fitToWindow(ecg: FloatArray): FloatArray {
    val window = FloatArray(INPUT_SAMPLES)
    val copyLength = minOf(ecg.size, INPUT_SAMPLES)
    System.arraycopy(ecg, 0, window, 0, copyLength)
    return window
  }

  private fun meanPool(tokens: Array<FloatArray>): FloatArray {
    require(tokens.isNotEmpty()) { "Encoder returned no tokens." }

    val embeddingDim = tokens[0].size
    val pooled = FloatArray(embeddingDim)
    for (token in tokens) {
      require(token.size == embeddingDim) { "Inconsistent token embedding size." }
      for (i in 0 until embeddingDim) {
        pooled[i] += token[i]
      }
    }
    for (i in 0 until embeddingDim) {
      pooled[i] /= tokens.size.toFloat()
    }
    return pooled
  }

  override fun close() {
    session.close()
  }

  companion object {
    private const val INPUT_NAME = "ecg"
    private const val SAMPLE_RATE_HZ = 500
    private const val WINDOW_SECONDS = 10
    private const val INPUT_SAMPLES = SAMPLE_RATE_HZ * WINDOW_SECONDS
  }
}
