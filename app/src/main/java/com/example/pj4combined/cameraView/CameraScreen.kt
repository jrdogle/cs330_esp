package com.example.pj4combined.cameraView

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.withStyle
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import com.example.pj4combined.cameraInference.DetectionResult
import com.example.pj4combined.cameraInference.DetectorListener
import com.example.pj4combined.cameraInference.PersonClassifier
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.util.concurrent.Executors
import kotlin.coroutines.resume
import kotlin.coroutines.suspendCoroutine


private lateinit var bitmapBuffer: Bitmap



@Composable
fun CameraScreen() {
    // Obtain the current context and lifecycle owner
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val detectionResults = remember { mutableStateOf<DetectionResult?>(null) }
    val errorMessage = remember { mutableStateOf<String?>(null) }

    val listener = remember {
        DetectorListener(
            onDetectionResult = { results -> detectionResults.value = results },
            onDetectionError = { error -> errorMessage.value = error }
        )
    }

    // Initialize our background executor
    val cameraExecutor = remember { Executors.newSingleThreadExecutor() }

    val resolutionSelector =  ResolutionSelector.Builder().setAspectRatioStrategy(AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY).build()
    val preview = remember { Preview.Builder().setResolutionSelector(resolutionSelector).build() }
    val imageAnalyzer = remember {ImageAnalysis.Builder()
            .setResolutionSelector(resolutionSelector)
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setOutputImageFormat(OUTPUT_IMAGE_FORMAT_RGBA_8888)
            .build() }


    val previewView = remember { PreviewView(context) }

    var currentClassifier by remember { mutableStateOf<PersonClassifier?>(null) }

    LaunchedEffect(Unit) {

        val personClassifierGPU = PersonClassifier()
        withContext(Dispatchers.IO) {
            personClassifierGPU.initialize(context, useGPU = true)
        }
        personClassifierGPU.setDetectorListener(listener)

        currentClassifier = personClassifierGPU

        preview.surfaceProvider = previewView.surfaceProvider

        // The analyzer can then be assigned to the instance
        imageAnalyzer.setAnalyzer(cameraExecutor) { image ->
            detectObjects(image, personClassifierGPU)
            // Close the image proxy
            image.close()
        }

        previewView.setScaleType(PreviewView.ScaleType.FIT_CENTER)

        val cameraProvider = context.getCameraProvider()
        cameraProvider.unbindAll()
        cameraProvider.bindToLifecycle(
            lifecycleOwner,
            CameraSelector.Builder().requireLensFacing(CameraSelector.LENS_FACING_FRONT).build(),
            preview,
            imageAnalyzer
        )
        Log.d("CS330", "Camera bound")
    }

    // Helper Functions
    suspend fun switchToCPUClassifier(
        context: Context,
        imageAnalyzer: ImageAnalysis,
        cameraExecutor: java.util.concurrent.ExecutorService,
        listener: DetectorListener
    ) {
        val personClassifierCPU = PersonClassifier()
        withContext(Dispatchers.IO) {
            personClassifierCPU.initialize(context, useGPU = false, threadNumber = 2)
        }
        personClassifierCPU.setDetectorListener(listener)

        imageAnalyzer.clearAnalyzer()
        imageAnalyzer.setAnalyzer(cameraExecutor) { image ->
            detectObjects(image, personClassifierCPU)
            image.close()
        }
        Log.d("CS330", "Switched to CPU for Object Detection")
    }

    suspend fun switchToGPUClassifier(
        context: Context,
        imageAnalyzer: ImageAnalysis,
        cameraExecutor: java.util.concurrent.ExecutorService,
        listener: DetectorListener
    ) {
        val personClassifierGPU = PersonClassifier()
        withContext(Dispatchers.IO) {
            personClassifierGPU.initialize(context, useGPU = true)
        }
        personClassifierGPU.setDetectorListener(listener)

        imageAnalyzer.clearAnalyzer()
        imageAnalyzer.setAnalyzer(cameraExecutor) { image ->
            detectObjects(image, personClassifierGPU)
            image.close()
        }
        Log.d("CS330", "Switched to GPU for Object Detection")
    }


    val currentMode = remember { mutableStateOf("GPU") } // "GPU" 또는 "CPU" 상태 추적
    val stableInferenceCount = remember { mutableIntStateOf(0) } // CPU에서의 안정적 성능 카운트

    if (detectionResults.value != null) {
        val inferenceTimeThreshold = 150
        val stableInferenceLimit = 100 // Number of stable cycles to switch back to GPU

        if (currentMode.value == "GPU" && detectionResults.value!!.inferenceTime > inferenceTimeThreshold) {
            // Switch to CPU
            LaunchedEffect(Unit) {
                switchToCPUClassifier(context, imageAnalyzer, cameraExecutor, listener)
                currentMode.value = "CPU"
                stableInferenceCount.intValue = 0
            }
        } else if (currentMode.value == "CPU") {
            if (detectionResults.value!!.inferenceTime <= inferenceTimeThreshold) {
                stableInferenceCount.intValue += 1
            } else {
                stableInferenceCount.intValue = 0
            }
            if (stableInferenceCount.intValue >= stableInferenceLimit) {
                // Switch back to GPU
                LaunchedEffect(Unit) {
                    switchToGPUClassifier(context, imageAnalyzer, cameraExecutor, listener)
                    currentMode.value = "GPU"
                }
            }
        }
    }

    // Display the Camera Preview
    ShowCameraPreview( previewView )
    Text(buildAnnotatedString {
            if (detectionResults.value != null) {
                val detectionResult = detectionResults.value!!
                withStyle(style = SpanStyle(color = Color.Blue, fontSize = 20.sp)) {
                    append("inference time ${detectionResult.inferenceTime}\n")
                }
                if (detectionResult.detections.find { it.categories[0].label == "person" } != null) {
                    withStyle(style = SpanStyle(color = Color.Red, fontSize = 40.sp)) {
                        append("human detected")
                    }
                }
            }
    })
}

@Composable
fun ShowCameraPreview(previewView: PreviewView) {
    AndroidView(
        { previewView },
        modifier = Modifier.fillMaxSize()
    )
}

fun detectObjects(image: ImageProxy, personClassifier: PersonClassifier) {
    if (!::bitmapBuffer.isInitialized) {
        // The image rotation and RGB image buffer are initialized only once
        // the analyzer has started running
        bitmapBuffer = Bitmap.createBitmap(
            image.width,
            image.height,
            Bitmap.Config.ARGB_8888
        )
    }
    // Copy out RGB bits to the shared bitmap buffer
    image.use { bitmapBuffer.copyPixelsFromBuffer(image.planes[0].buffer) }
    val imageRotation = image.imageInfo.rotationDegrees

    // Pass Bitmap and rotation to the object detector helper for processing and detection
    personClassifier.detect(bitmapBuffer, imageRotation)
}

suspend fun Context.getCameraProvider(): ProcessCameraProvider = suspendCoroutine { continuation ->
    ProcessCameraProvider.getInstance(this).also { cameraProvider ->
        cameraProvider.addListener({
            continuation.resume(cameraProvider.get())
        }, ContextCompat.getMainExecutor(this))
    }
}