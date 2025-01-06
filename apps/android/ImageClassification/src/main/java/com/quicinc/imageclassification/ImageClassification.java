// ---------------------------------------------------------------------
// Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// ---------------------------------------------------------------------
package com.quicinc.imageclassification;

import android.content.Context;
import android.graphics.Bitmap;
import android.util.Log;
import android.util.Pair;

import com.quicinc.ImageProcessing;
import com.quicinc.tflite.AIHubDefaults;
import com.quicinc.tflite.TFLiteHelpers;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Delegate;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.Tensor;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;

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

public class ImageClassification implements AutoCloseable {
    private static final int DEFAULT_TOP_K = 5;
    private static final String TAG = "ImageClassification";
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
    public ImageClassification(Context context,
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
    public ImageClassification(Context context,
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
        assert tfLiteInterpreter.getInputTensorCount() == 1;
        Tensor inputTensor = tfLiteInterpreter.getInputTensor(0);
        inputShape = inputTensor.shape();
        inputType = inputTensor.dataType();
        assert inputShape.length == 4; // 4D Input Tensor: [Batch, Height, Width, Channels]
        assert inputShape[0] == 1; // Batch size is 1
        assert inputShape[1] == 3; // Input tensor should have 3 channels
        assert inputType == DataType.UINT8 || inputType == DataType.FLOAT32; // INT8 (Quantized) and FP32 Input Supported

        assert tfLiteInterpreter.getOutputTensorCount() == 1;
        Tensor outputTensor = tfLiteInterpreter.getOutputTensor(0);
        int[] outputShape = outputTensor.shape();
        outputType = outputTensor.dataType();
        assert outputShape.length == 2; // 2D Output Tensor: [Batch, # of Labels]
        assert inputShape[0] == 1; // Batch size is 1
        // 여기서 outputShape이... 976

        assert outputShape[1] == labelList.size(); // # of labels == output dim
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
    private ByteBuffer[] preprocess(Bitmap image) {
        long prepStartTime = System.nanoTime();
        Bitmap resizedImg;

        // Resize input image
        if (image.getHeight() != inputShape[2] || image.getWidth() != inputShape[3]) {
            resizedImg = ImageProcessing.resizeAndPadMaintainAspectRatio(image, inputShape[2], inputShape[3], 0);
        } else {
            resizedImg = image;
        }

        // Convert type and fill input buffer
        ByteBuffer inputBuffer;
        TensorImage tImg = TensorImage.fromBitmap(resizedImg);
        if (inputType == DataType.FLOAT32) {
            inputBuffer = imageProcessor.process(tImg).getBuffer();
        } else {
            inputBuffer = tImg.getTensorBuffer().getBuffer();
        }

        preprocessingTime = System.nanoTime() - prepStartTime;
        Log.d(TAG, "Preprocessing Time: " + preprocessingTime / 1000000 + " ms");

        return new ByteBuffer[] {inputBuffer};
    }

    public ByteBuffer[] preprocessImage(Bitmap image) {
        long prepStartTime = System.nanoTime();

        // 1. 이미지 리사이즈 (480x480 고정)
        Bitmap resizedImage = Bitmap.createScaledBitmap(image, 480, 480, true);

        // 2. ByteBuffer 생성 (FLOAT32, 1x3x480x480)
        ByteBuffer inputBuffer = ByteBuffer.allocateDirect(4 * 480 * 480 * 3); // 4 bytes per float
        inputBuffer.order(ByteOrder.nativeOrder());

        // 3. 채널 우선 순서로 픽셀 데이터 저장 (1, 3, 480, 480)
        for (int c = 0; c < 3; c++) { // 채널 순서: R, G, B
            for (int y = 0; y < 480; y++) {
                for (int x = 0; x < 480; x++) {
                    int pixel = resizedImage.getPixel(x, y);

                    float value;
                    if (c == 0) { // R 채널
                        value = ((pixel >> 16) & 0xFF) / 255.0f;
                    } else if (c == 1) { // G 채널
                        value = ((pixel >> 8) & 0xFF) / 255.0f;
                    } else { // B 채널
                        value = (pixel & 0xFF) / 255.0f;
                    }

                    // Normalize: [-1, 1] 범위로 이동
                    value = (value - 0.5f) / 0.5f;

                    // ByteBuffer에 저장
                    inputBuffer.putFloat(value);
                }
            }
        }

        // 4. 전처리 시간 로깅

        preprocessingTime = System.nanoTime() - prepStartTime;
        Log.d(TAG, "Preprocessing Time: " + preprocessingTime / 1000000 + " ms");

        // 5. ByteBuffer 반환
        return new ByteBuffer[] {inputBuffer};
    }

    /**
     * Reads the output buffers on tfLiteModel and processes them into readable output classes.
     *
     * @return Predicted object class names, in order of confidence (highest confidence first).
     */
    private ArrayList<String> postprocess() {
        long postStartTime = System.nanoTime();

        List<Integer> indexList;
        ByteBuffer outputBuffer = tfLiteInterpreter.getOutputTensor(0).asReadOnlyBuffer();
        if (outputType == DataType.FLOAT32) {
            indexList = findTopKFloatIndices(outputBuffer.asFloatBuffer(), TOPK);
        } else {
            indexList = findTopKByteIndices(outputBuffer, TOPK);
        }
        ArrayList<String> labels = indexList.stream().map(labelList::get).collect(Collectors.toCollection(ArrayList<String>::new));

        postprocessingTime = System.nanoTime() - postStartTime;
        Log.d(TAG, "Postprocessing Time: " + postprocessingTime / 1000000 + " ms");

        return labels;
    }

    private ArrayList<Pair<String, Float>> postprocessSoftmaxTopK(int topK) {
        long postStartTime = System.nanoTime();

        // 1. 출력 텐서 가져오기
        ByteBuffer outputBuffer = tfLiteInterpreter.getOutputTensor(0).asReadOnlyBuffer();
        FloatBuffer probabilitiesBuffer = outputBuffer.asFloatBuffer();

        // 2. 확률 값을 Float 배열로 변환
        float[] probabilities = new float[probabilitiesBuffer.capacity()];
        probabilitiesBuffer.get(probabilities);

        // 3. Softmax 적용 (TFLite 모델이 Softmax를 이미 적용했을 경우 생략 가능)
        float sum = 0f;
        for (float prob : probabilities) {
            sum += (float) Math.exp(prob);
        }
        for (int i = 0; i < probabilities.length; i++) {
            probabilities[i] = (float) Math.exp(probabilities[i]) / sum;
        }

        // 4. 상위 K개의 클래스와 확률 추출
        PriorityQueue<Pair<Integer, Float>> pq = new PriorityQueue<>(
                (a, b) -> Float.compare(b.second, a.second)
        );
        for (int i = 0; i < probabilities.length; i++) {
            pq.add(new Pair<>(i, probabilities[i]));
        }

        ArrayList<Pair<String, Float>> results = new ArrayList<>();
        for (int i = 0; i < topK && !pq.isEmpty(); i++) {
            Pair<Integer, Float> top = pq.poll();
            results.add(new Pair<>(labelList.get(top.first), top.second)); // 클래스 이름 매핑
        }

        postprocessingTime = System.nanoTime() - postStartTime;
        Log.d(TAG, "Postprocessing Time: " + postprocessingTime / 1000000 + " ms");

        // 5. 결과 반환
        return results;
    }


    /**
     * Predict the most likely classes of the object in the image.
     *
     * @param image RGBA-8888 bitmap image to predict class of.
     * @return Predicted object class names, in order of confidence (highest confidence first).
     */
    public ArrayList<String> predictClassesFromImage(Bitmap image) {
        // Preprocessing: Resize, convert type
//        ByteBuffer[] inputs = preprocess(image);
        ByteBuffer[] inputs = preprocessImage(image);

        // Inference
        tfLiteInterpreter.runForMultipleInputsOutputs(inputs, new HashMap<>());

        // tmp
        // Postprocessing: Compute top K indices and convert to labels
        ArrayList<Pair<String, Float>> results = postprocessSoftmaxTopK(DEFAULT_TOP_K);

        // 확률값(Float)을 제외하고 클래스 이름(String)만 반환
        ArrayList<String> classNames = new ArrayList<>();
        for (Pair<String, Float> result : results) {
            classNames.add(result.first); // 클래스 이름 추가

            Log.d(TAG, "Class: " + result.first + ", Score: " + result.second);
        }
        //

        // Postprocessing: Compute top K indices and convert to labels
        return classNames;
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
