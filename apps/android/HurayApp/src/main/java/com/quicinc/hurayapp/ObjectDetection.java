// ---------------------------------------------------------------------
// Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// ---------------------------------------------------------------------
package com.quicinc.hurayapp;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.util.Log;
import android.util.Pair;

import com.quicinc.tflite.AIHubDefaults;
import com.quicinc.tflite.TFLiteHelpers;

import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfRect2d;
import org.opencv.core.Rect2d;
import org.opencv.dnn.Dnn;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Delegate;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.Tensor;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.MappedByteBuffer;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.stream.Collectors;


public class ObjectDetection implements AutoCloseable {
    private static final String TAG = "ObjectDetection";
    private final Interpreter tfLiteInterpreter;
    private final Map<TFLiteHelpers.DelegateType, Delegate> tfLiteDelegateStore;
    private final List<String> labelList;
    private final int[] inputShape;
    private final DataType inputType;
    private final DataType outputType;
    private long preprocessingTime;
    private long postprocessingTime;
    private static final int TOPK = 3;
    private final ImageProcessor imageProcessor;

    private static final float confThreshold = 0.25f; // Confidence threshold
    private static final float iouThreshold = 0.5f;  // IOU threshold


    /**
     * Create an Image Classifier from the given model.
     * Uses default compute units: NPU, GPU, CPU.
     * Ignores compute units that fail to load.
     *
     * @param context    App context.
     * @param modelPath  Model path to load.
     * @param labelsPath Labels path to load.
     * @throws IOException If the model can't be read from disk.
     */
    public ObjectDetection(Context context,
                           String modelPath,
                           String labelsPath) throws IOException, NoSuchAlgorithmException {
        this(context, modelPath, labelsPath, AIHubDefaults.delegatePriorityOrder);
    }

    /**
     * Create an Image Classifier from the given model.
     * Ignores compute units that fail to load.
     *
     * @param context     App context.
     * @param modelPath   Model path to load.
     * @param labelsPath  Labels path to load.
     * @param delegatePriorityOrder Priority order of delegate sets to enable.
     * @throws IOException If the model can't be read from disk.
     */
    public ObjectDetection(Context context,
                           String modelPath,
                           String labelsPath,
                           TFLiteHelpers.DelegateType[][] delegatePriorityOrder) throws IOException, NoSuchAlgorithmException {
        // Load labels
        try (BufferedReader labelsFile = new BufferedReader(new InputStreamReader(context.getAssets().open(labelsPath)))) {
            labelList = labelsFile.lines().collect(Collectors.toCollection(ArrayList::new));
        }

        // Load TF Lite model
        Pair<MappedByteBuffer, String> modelAndHash = TFLiteHelpers.loadModelFile(context.getAssets(), modelPath);
        Pair<Interpreter, Map<TFLiteHelpers.DelegateType, Delegate>> iResult = TFLiteHelpers.CreateInterpreterAndDelegatesFromOptions(
            modelAndHash.first,
            delegatePriorityOrder,
            AIHubDefaults.numCPUThreads,
            context.getApplicationInfo().nativeLibraryDir,
            context.getCacheDir().getAbsolutePath(),
            modelAndHash.second
        );
        tfLiteInterpreter = iResult.first;
        tfLiteDelegateStore = iResult.second;

        // Validate TF Lite model fits requirements for this app

        // TODO: check shape
        assert tfLiteInterpreter.getInputTensorCount() == 1;
        Tensor inputTensor = tfLiteInterpreter.getInputTensor(0);
        inputShape = inputTensor.shape();
        inputType = inputTensor.dataType();

        assert inputShape.length == 4; // 4D Input Tensor: [Batch, Height, Width, Channels]
        assert inputShape[0] == 1; // Batch size is 1
        assert inputShape[3] == 3; // Input tensor should have 3 channels
        assert inputType == DataType.UINT8 || inputType == DataType.FLOAT32; // INT8 (Quantized) and FP32 Input Supported

        assert tfLiteInterpreter.getOutputTensorCount() == 1;
        Tensor outputTensor = tfLiteInterpreter.getOutputTensor(0);
        int[] outputShape = outputTensor.shape();
        outputType = outputTensor.dataType();

        assert outputShape.length == 3; // 2D Output Tensor: [Batch, 6, pred boxes]
        assert inputShape[0] == 1; // Batch size is 1

        assert outputType == DataType.UINT8 || outputType == DataType.INT8 | outputType == DataType.FLOAT32; // U/INT8 (Quantized) and FP32 Output Supported

        // Set-up preprocessor
        imageProcessor = new ImageProcessor.Builder().add(new NormalizeOp(0.0f, 255.0f)).build();
    }

    /**
     * Free resources used by the classifier.
     */
    @Override
    public void close() {
        tfLiteInterpreter.close();
        for (Delegate delegate: tfLiteDelegateStore.values()) {
            delegate.close();
        }
    }

    /**
     * @return last preprocessing time in microseconds.
     */
    public long getLastPreprocessingTime() {
        if (preprocessingTime == 0) {
            throw new RuntimeException("Cannot get preprocessing time as model has not yet been executed.");
        }
        return preprocessingTime;
    }

    /**
     * @return last inference time in microseconds.
     */
    public long getLastInferenceTime() {
        return tfLiteInterpreter.getLastNativeInferenceDurationNanoseconds();
    }

    /**
     * @return last postprocessing time in microseconds.
     */
    public long getLastPostprocessingTime() {
        if (postprocessingTime == 0) {
            throw new RuntimeException("Cannot get postprocessing time as model has not yet been executed.");
        }
        return postprocessingTime;
    }

    /**
     * Preprocess using the provided image (resize, convert to model input data type).
     * Sets the input buffer held by this.tfLiteModel to the processed input.
     *
     * @param image RGBA-8888 Bitmap to preprocess.
     * @return Array of inputs to pass to the interpreter.
     */
    private ResizeResult preprocess(Bitmap image) {
        long prepStartTime = System.nanoTime(); // Start time for preprocessing

        // Resize and pad while maintaining aspect ratio
        ResizeResult result = letterbox(image, inputShape[1], inputShape[2]);

        // Calculate and log preprocessing time
        preprocessingTime = System.nanoTime() - prepStartTime;
        Log.d(TAG, "Preprocessing Time: " + preprocessingTime / 1_000_000 + " ms");

        return result;
    }

    // Helper class to store resize results
    private static class ResizeResult {
        Bitmap processedImage;
        float[] padding;

        ResizeResult(Bitmap processedImage, float[] padding) {
            this.processedImage = processedImage;
            this.padding = padding;
        }
    }

    private ResizeResult letterbox(Bitmap img, int newWidth, int newHeight) {
        int originalWidth = img.getWidth();
        int originalHeight = img.getHeight();

        // Calculate scaling ratio
        float scale = Math.min(newWidth / (float) originalWidth, newHeight / (float) originalHeight);

        // Compute new dimensions
        int newUnpadWidth = Math.round(originalWidth * scale);
        int newUnpadHeight = Math.round(originalHeight * scale);

        // Compute padding
        int dw = (newWidth - newUnpadWidth) / 2;
        int dh = (newHeight - newUnpadHeight) / 2;

        // Resize and pad the image
        Bitmap resizedImage = Bitmap.createScaledBitmap(img, newUnpadWidth, newUnpadHeight, true);
        Bitmap paddedImage = Bitmap.createBitmap(newWidth, newHeight, Bitmap.Config.ARGB_8888);
        Canvas canvas = new Canvas(paddedImage);
        canvas.drawColor(Color.rgb(114, 114, 114)); // Fill with padding color
        canvas.drawBitmap(resizedImage, dw, dh, null);

        return new ResizeResult(paddedImage, new float[] {dh / (float) newHeight, dw / (float) newWidth});
    }

    private ByteBuffer convertBitmapToByteBuffer(Bitmap bitmap, boolean normalize) {
        int width = bitmap.getWidth();
        int height = bitmap.getHeight();
        int numChannels = 3;
        ByteBuffer buffer = ByteBuffer.allocateDirect(width * height * numChannels * (normalize ? 4 : 1));
        buffer.order(ByteOrder.nativeOrder());

        int[] pixels = new int[width * height];
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height);

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int pixel = pixels[y * width + x];

                // Extract RGB values
                int r = (pixel >> 16) & 0xFF;
                int g = (pixel >> 8) & 0xFF;
                int b = pixel & 0xFF;

                if (normalize) {
                    buffer.putFloat(r / 255.0f);
                    buffer.putFloat(g / 255.0f);
                    buffer.putFloat(b / 255.0f);
                } else {
                    buffer.put((byte) r);
                    buffer.put((byte) g);
                    buffer.put((byte) b);
                }
            }
        }

        buffer.rewind();
        return buffer;
    }

    /**
     * Reads the output buffers on tfLiteModel and processes them into readable output classes.
     *
     * @return Predicted object class names, in order of confidence (highest confidence first).
     */

    private List<Rect2d> postprocess(Bitmap img, float[][][] outputs, float[] padding) {
        long postStartTime = System.nanoTime();

        List<Rect2d> boxes = new ArrayList<>();
        List<Float> scores = new ArrayList<>();

        List<Rect2d> filteredBoxes = new ArrayList<>();

        // 이미지 크기 및 스케일 계산
        int imgWidth = img.getWidth();
        int imgHeight = img.getHeight();
        float maxDimension = Math.max(imgWidth, imgHeight);

        // 모델 출력 데이터를 변환
        for (int i = 0; i < outputs[0][0].length; i++) {
            // 모델 출력값 (cx, cy, w, h, confidence)
            float cx = outputs[0][0][i];
            float cy = outputs[0][1][i];
            float w = outputs[0][2][i];
            float h = outputs[0][3][i];
            float confidence = outputs[0][4][i];

            // Confidence Threshold 적용
            if (confidence > confThreshold) {

                // 패딩 및 크기 조정
                cx = (cx - padding[1]) * maxDimension;
                cy = (cy - padding[0]) * maxDimension;
                w *= maxDimension;
                h *= maxDimension;

                // 좌표 변환: (cx, cy, w, h) -> (xmin, ymin, xmax, ymax)
                double xmin = cx - w / 2.0;
                double ymin = cy - h / 2.0;
                double xmax = cx + w / 2.0;
                double ymax = cy + h / 2.0;

                // 박스 정보 저장
                boxes.add(new Rect2d(xmin, ymin, xmax - xmin, ymax - ymin));
                scores.add(confidence);

            }
        }
        // NMS 적용
        if (!boxes.isEmpty() && !scores.isEmpty()) {
            MatOfRect2d matBoxes = new MatOfRect2d();
            matBoxes.fromList(boxes);

            MatOfFloat matScores = new MatOfFloat();
            matScores.fromArray(toPrimitive(scores));

            MatOfInt nmsIndices = new MatOfInt();
            Dnn.NMSBoxes(matBoxes, matScores, confThreshold, iouThreshold, nmsIndices);
            // TODO: nmsIndices만 가지고 bbox를 return

            for (int idx : nmsIndices.toArray()) {
                Rect2d box = boxes.get(idx);
                filteredBoxes.add(box);
            }
        }

        postprocessingTime = System.nanoTime() - postStartTime;
        Log.d(TAG, "Postprocessing Time: " + postprocessingTime / 1_000_000 + " ms");

        return filteredBoxes;
    }

    // float[] 변환 유틸리티 함수
    private float[] toPrimitive(List<Float> list) {
        float[] array = new float[list.size()];
        for (int i = 0; i < list.size(); i++) {
            array[i] = list.get(i);
        }
        return array;
    }

    /**
     * Predict the most likely classes of the object in the image.
     *
     * @param image RGBA-8888 bitmap image to predict class of.
     * @return Predicted object class names, in order of confidence (highest confidence first).
     */
    public List<Rect2d> detectObjectsFromImage(Bitmap image) {
        // Preprocessing: Resize, convert type
        ResizeResult result = preprocess(image);
        Bitmap paddedImage = result.processedImage;
        float[] padding = result.padding;

        // Convert padded image to ByteBuffer for model input
        ByteBuffer inputBuffer;
        if (inputType == DataType.FLOAT32) {
            inputBuffer = convertBitmapToByteBuffer(paddedImage, true); // Normalize to [0, 1]
        } else {
            inputBuffer = convertBitmapToByteBuffer(paddedImage, false); // keep values as UINT8
        }

        // Inference
        Map<Integer, Object> outputs = new HashMap<>();
        float[][][] outputBuffer = new float[1][6][8400]; // YOLO 모델 출력 크기 (예시)
        outputs.put(0, outputBuffer);
        tfLiteInterpreter.runForMultipleInputsOutputs(new ByteBuffer[]{inputBuffer}, outputs);

        // Postprocessing: Process the outputs and return bboxes
        return postprocess(image, outputBuffer, padding);
    }

    /**
     * Return the indices of the top K elements in a float buffer.
     *
     * @param fb The float buffer to read values from.
     * @param k The number of indices to return.
     * @return The indices of the top K elements in the buffer.
     */
    private static List<Integer> findTopKFloatIndices(FloatBuffer fb, int k) {
        class ValueAndIdx implements Comparable<ValueAndIdx>{
            public float value;
            public int idx;

            @Override public int compareTo(ValueAndIdx other) {
                return Float.compare(value, other.value);
            }

            public ValueAndIdx(float value, int idx) {
                this.value = value;
                this.idx = idx;
            }
        }

        PriorityQueue<ValueAndIdx> maxHeap = new PriorityQueue<>();
        int i = 0;
        while (fb.hasRemaining()) {
            maxHeap.add(new ValueAndIdx(fb.get(), i));
            if (maxHeap.size() > k) {
                maxHeap.poll();
            }
            i++;
        }

        ArrayList<Integer> topKList = maxHeap.stream().map(x -> x.idx).collect(Collectors.toCollection(ArrayList::new));
        Collections.reverse(topKList);
        return topKList;
    }

    /**
     * Return the indices of the top K elements in a byte buffer.
     *
     * @param bb The byte buffer to read values from.
     * @param k The number of indices to return.
     * @return The indices of the top K elements in the buffer.
     */
    private static List<Integer> findTopKByteIndices(ByteBuffer bb, int k) {
        class ValueAndIdx implements Comparable<ValueAndIdx>{
            public byte value;
            public int idx;

            @Override public int compareTo(ValueAndIdx other) {
                return Byte.compare(value, other.value);
            }

            public ValueAndIdx(byte value, int idx) {
                this.value = value;
                this.idx = idx;
            }
        }

        PriorityQueue<ValueAndIdx> maxHeap = new PriorityQueue<>();
        int i = 0;
        while (bb.hasRemaining()) {
            maxHeap.add(new ValueAndIdx(bb.get(), i));
            if (maxHeap.size() > k) {
                maxHeap.poll();
            }
            i++;
        }

        ArrayList<Integer> topKList = maxHeap.stream().map(x -> x.idx).collect(Collectors.toCollection(ArrayList::new));
        Collections.reverse(topKList);
        return topKList;
    }
}
