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

//import com.quicinc.tflite.AIHubDefaults;
import com.quicinc.tflite.ClsAIHubDefaults;
import com.quicinc.tflite.TFLiteHelpers;

import org.opencv.core.Rect2d;
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

public class ImageClassification implements AutoCloseable {
    private static final int DEFAULT_TOP_K = 5;
    private static final String TAG = "HurayFoodClassifier";
    private final Interpreter tfLiteInterpreter;
    private final Map<TFLiteHelpers.DelegateType, Delegate> tfLiteDelegateStore;
    private final List<String> labelList;
    private final int[] inputShape;
    private final DataType inputType;
    private final DataType outputType;
    private long preprocessingTime;
    private long postprocessingTime;
    private List<Long> infTimeList;
    private List<Long> postprocessTimeList;
    private List<Long> preprocessTimeList;
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
        this(context, modelPath, labelsPath, ClsAIHubDefaults.delegatePriorityOrder);
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
            ClsAIHubDefaults.numCPUThreads,
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

    public List<Long> getTotalInfTimeList() {
        if (infTimeList.size() == 0) {
            throw new RuntimeException("Cannot detect any food in the image.");
        }
        return infTimeList;
    }

    public double getAverageInferenceTime() {
        long total = 0;
        for (long time : infTimeList) {
            total += time;
        }
        return (double) total / infTimeList.size();
    }

    public double getTotalInfTime() {
        long total = 0;
        for (long time : infTimeList) {
            total += time;
        }
        return (double) total;
    }

    public double getTotalPreprocessTime() {
        long total = 0;
        for (long time : preprocessTimeList) {
            total += time;
        }
        return (double) total;
    }

    public double getTotalPostprocesstime() {
        long total = 0;
        for (long time : postprocessTimeList) {
            total += time;
        }
        return (double) total;
    }

    /**
     * Preprocess using the provided image (resize, convert to model input data type).
     * Sets the input buffer held by this.tfLiteModel to the processed input.
     *
     * @param image RGBA-8888 Bitmap to preprocess.
     * @return Array of inputs to pass to the interpreter.
     */

    public ByteBuffer[] preprocess(Bitmap image) {

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

        return new ByteBuffer[] {inputBuffer};
    }

    /**
     * Reads the output buffers on tfLiteModel and processes them into readable output classes.
     *
     * @return Predicted object class names, in order of confidence (highest confidence first).
     */

    private ArrayList<Pair<String, Float>> postprocess(int topK) {

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

        // 5. 결과 반환
        return results;
    }


    /**
     * Predict the most likely classes of the object in the image.
     *
     * @param image RGBA-8888 bitmap image to predict class of.
     * @return Predicted object class names, in order of confidence (highest confidence first).
     */
    public Pair<Bitmap, ArrayList<String>> predictClassesFromImage(Bitmap image, List<Rect2d> boxes) {

        ArrayList<String> classNames = new ArrayList<>();
        infTimeList = new ArrayList<>();
        postprocessTimeList = new ArrayList<>();
        preprocessTimeList = new ArrayList<>();

        int idx = 1;

        for (Rect2d box : boxes) {
            // TODO: preprocess..
            long prepStartTime = System.nanoTime();
            // Bounding Box 좌표
            int xmin = (int) Math.max(0, box.x);
            int ymin = (int) Math.max(0, box.y);
            int xmax = (int) Math.min(image.getWidth(), box.x + box.width);
            int ymax = (int) Math.min(image.getHeight(), box.y + box.height);

            int width = xmax - xmin;
            int height = ymax - ymin;

            // 잘못된 크기(음수 또는 0 크기) 인 경우 스킵
            if (width <= 0 || height <= 0) {
                continue;
            }

            // Bitmap에서 Crop
            Bitmap croppedImage = Bitmap.createBitmap(image, xmin, ymin, width, height);

            // 리스트에 추가
            ByteBuffer[] inputs = preprocess(croppedImage);
            preprocessingTime = System.nanoTime() - prepStartTime;
            Log.d("debugInferenceTime", "cls Preprocessing Time: " + preprocessingTime / 1000000 + " ms");
            preprocessTimeList.add(preprocessingTime/1000000);


            // Inference
            /* TODO: check inferencetime
                여기서 체크하는게, 마지막에 lastinference로 불러오는 시간과 같은지 체크해야할 것 같다.
                imageClassification.getLastInferenceTime();
                마지막 Inference한 것을 가져오는 것으로 보이는데...
            */
            tfLiteInterpreter.runForMultipleInputsOutputs(inputs, new HashMap<>());
            infTimeList.add(tfLiteInterpreter.getLastNativeInferenceDurationNanoseconds() / 1000000);

            //Postprocessing: Compute top K indices and convert to labels
            long postStartTime = System.nanoTime();
            ArrayList<Pair<String, Float>> results = postprocess(DEFAULT_TOP_K);
            postprocessingTime = System.nanoTime() - postStartTime;
            Log.d("debugInferenceTime", "cls Postprocessing Time: " + postprocessingTime / 1000000 + " ms");
            postprocessTimeList.add(postprocessingTime/1000000);

            ArrayList<String> innerClassNames = new ArrayList<>();

            // 확률값(Float)를 제외하고 클래스 이름(String)만 반환
            for (Pair<String, Float> result : results) {
                innerClassNames.add(result.first); // 클래스 이름 추가

                Log.d(TAG, "Class: " + result.first + ", Score: " + result.second);
            }

            String joinedLabels = String.join(", ", innerClassNames);
            classNames.add(idx + ": " + joinedLabels);
            idx++;
        }
        Log.d("debugInferenceTime", "cls Total preprocess time list: " + preprocessTimeList);
        Log.d("debugInferenceTime", "cls Total inference time list: " + infTimeList);
        Log.d("debugInferenceTime", "cls Total postprocess time list: " + postprocessTimeList);
        Bitmap processedImage = drawBbox(image, boxes, classNames);
        return new Pair<> (processedImage, classNames);
    }


    private static Bitmap drawBbox(Bitmap image, List<Rect2d> boxes, ArrayList<String> predResults) {

        int imgWidth = image.getWidth();
        int imgHeight = image.getHeight();

        Paint paint = new Paint();
        paint.setColor(Color.RED);
        paint.setStyle(Paint.Style.STROKE);
        paint.setStrokeWidth(2);

        Paint textPaint = new Paint();
        textPaint.setColor(Color.WHITE);
        textPaint.setTextSize(30);

        Bitmap mutableImage = image.copy(Bitmap.Config.ARGB_8888, true);
        Canvas canvas = new Canvas(mutableImage);

        int idx = 0;

        for (Rect2d box : boxes) {
            canvas.drawRect(
                    (float) box.x,
                    (float) box.y,
                    (float) (box.x + box.width),
                    (float) (box.y + box.height),
                    paint
            );

            String text = predResults.get(idx);

            float textWidth = textPaint.measureText(text);
            float textHeight = textPaint.getTextSize();
            float labelX = Math.max(0, Math.min((float) box.x, imgWidth - textWidth));
            float labelY = Math.max(0, Math.min((float) box.y - 10, imgHeight - textHeight));

            canvas.drawText(
                    text,
                    labelX,
                    labelY + 35,
                    textPaint
            );
            idx++;
        }
        return mutableImage;
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
