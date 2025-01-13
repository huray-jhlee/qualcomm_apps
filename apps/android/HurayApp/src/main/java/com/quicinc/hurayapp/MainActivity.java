// ---------------------------------------------------------------------
// Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// ---------------------------------------------------------------------
package com.quicinc.hurayapp;

import android.app.Activity;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.ImageDecoder;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.provider.MediaStore;
import android.util.Pair;
import android.view.View;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.RadioButton;
import android.widget.RadioGroup;
import android.widget.Spinner;
import android.widget.TextView;
import android.util.Log;

import androidx.activity.result.ActivityResult;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AppCompatActivity;

import com.quicinc.ImageProcessing;
import com.quicinc.tflite.AIHubDefaults;
import com.quicinc.tflite.ClsAIHubDefaults;

import org.opencv.core.Rect2d;

import java.io.IOException;
import java.io.InputStream;
import java.security.NoSuchAlgorithmException;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import org.opencv.android.OpenCVLoader;


public class MainActivity extends AppCompatActivity {
    // UI Elements
    RadioGroup delegateSelectionGroup;
    RadioButton allDelegatesButton;
    RadioButton cpuOnlyButton;
    ImageView selectedImageView;
    TextView predictedClassesView;
    TextView detInferenceTimeView;
    TextView clsInferenceTimeView;
    TextView predictionTimeView;
    Spinner imageSelector;
    Button predictionButton;
    ActivityResultLauncher<Intent> selectImageResultLauncher;
    private final String fromGalleryImageSelectorOption = "From Gallery";
    private final String notSelectedImageSelectorOption = "Not Selected";
    private final String[] imageSelectorOptions =
            { notSelectedImageSelectorOption,
                    "Sample1.jpg",
                    "Sample2.jpg",
                    "Sample3.jpg",
                    fromGalleryImageSelectorOption};

    // Inference Elements
    Bitmap selectedImage = null; // Raw image, not resized
    private ImageClassification defaultDelegateClassifier;
    private ImageClassification cpuOnlyClassifier;

    // TODO: add object detection
    private ObjectDetection defaultDelegateDetector;
    private ObjectDetection cpuOnlyDetector;
    private boolean cpuOnlyInference = false;
    NumberFormat timeFormatter = new DecimalFormat("0.00");
    ExecutorService backgroundTaskExecutor = Executors.newSingleThreadExecutor();
    Handler mainLooperHandler = new Handler(Looper.getMainLooper());

    /**
     * Instantiate the activity on first load.
     * Creates the UI and a background thread that instantiates the classifier  TFLite model.
     *
     * @param savedInstanceState Saved instance state.
     */
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        // OpenCV 초기화
        if (!OpenCVLoader.initDebug()){
            Log.e("OpenCV", "OpenCV initialization failed!");
        } else {
            Log.d("OpenCV", "OpenCV initialization successful!");
        }

        //
        // UI Initialization
        //
        setContentView(R.layout.main_activity);
        selectedImageView = (ImageView) findViewById(R.id.selectedImageView);
        delegateSelectionGroup = (RadioGroup) findViewById(R.id.delegateSelectionGroup);
        cpuOnlyButton = (RadioButton)findViewById(R.id.cpuOnlyRadio);
        allDelegatesButton = (RadioButton)findViewById(R.id.defaultDelegateRadio);

        imageSelector = (Spinner) findViewById((R.id.imageSelector));
        predictedClassesView = (TextView)findViewById(R.id.predictionResultText);
        detInferenceTimeView = (TextView)findViewById(R.id.detInferenceTimeResultText);
        clsInferenceTimeView = (TextView)findViewById(R.id.clsInferenceTimeResultText);
        predictionTimeView = (TextView)findViewById(R.id.predictionTimeResultText);
        predictionButton = (Button)findViewById(R.id.runModelButton);

        // Setup Image Selector Dropdown
        ArrayAdapter ad = new ArrayAdapter(this, android.R.layout.simple_spinner_item, imageSelectorOptions);
        ad.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        imageSelector.setAdapter(ad);
        imageSelector.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> parent, View view, int position, long id) {
                // Load selected picture from assets
                ((TextView) view).setTextColor(getResources().getColor(R.color.white));
                if (!parent.getItemAtPosition(position).equals(notSelectedImageSelectorOption)) {
                    if (parent.getItemAtPosition(position).equals(fromGalleryImageSelectorOption)) {
                        Intent i = new Intent();
                        i.setType("image/*");
                        i.setAction(Intent.ACTION_GET_CONTENT);
                        selectImageResultLauncher.launch(i);
                    } else {
                        loadImageFromStringAsync((String) parent.getItemAtPosition(position));
                    }
                } else {
                    displayDefaultImage();
                }
            }

            @Override
            public void onNothingSelected(AdapterView<?> parent) { }
        });

        // Setup Image Selection from Phone Gallery
        selectImageResultLauncher = registerForActivityResult(
                new ActivityResultContracts.StartActivityForResult(),
                (ActivityResult result) -> {
                    if (result.getResultCode() == Activity.RESULT_OK &&
                            result.getData() != null &&
                            result.getData().getData() != null) {
                        loadImageFromURIAsync((Uri)(result.getData().getData()));
                    } else {
                        displayDefaultImage();
                    }
                });

        // Setup delegate selection buttons
        delegateSelectionGroup.setOnCheckedChangeListener((group, checkedId) -> {
            if (checkedId == R.id.cpuOnlyRadio) {
                if (!cpuOnlyInference) {
                    this.cpuOnlyInference = true;
                    clearPredictionResults();
                }
            } else if (checkedId == R.id.defaultDelegateRadio) {
                if (cpuOnlyInference) {
                    this.cpuOnlyInference = false;
                    clearPredictionResults();
                }
            } else {
                throw new RuntimeException("A radio button for selected runtime is not implemented");
            }
        });

        // Setup button callback
        predictionButton.setOnClickListener((view) -> updatePredictionDataAsync());

        // Exit the UI thread and instantiate the model in the background.
        createTFLiteModelAsync();

        // Enable image selection
        enableImageSelector();
        enableDelegateSelectionButtons();
    }

    /**
     * Enable or disable UI controls for inference.
     *
     * @param enabled If true, enable the UI. If false, disable the UI.
     */
    void setInferenceUIEnabled(boolean enabled) {
        if (!enabled) {
            predictedClassesView.setText("");
            detInferenceTimeView.setText("-- ms");
            clsInferenceTimeView.setText("-- ms");
            predictionTimeView.setText("-- ms");
            predictionButton.setEnabled(false);
            predictionButton.setAlpha(0.5f);
            imageSelector.setEnabled(false);
            imageSelector.setAlpha(0.5f);
            cpuOnlyButton.setEnabled(false);
            allDelegatesButton.setEnabled(false);
        } else if (cpuOnlyClassifier != null && cpuOnlyDetector != null && defaultDelegateClassifier != null && defaultDelegateDetector != null && selectedImage != null) {
            predictionButton.setEnabled(true);
            predictionButton.setAlpha(1.0f);
            enableImageSelector();
            enableDelegateSelectionButtons();
        }
    }

    /**
     * Enable the image selector UI spinner.
     */
    void enableImageSelector() {
        imageSelector.setEnabled(true);
        imageSelector.setAlpha(1.0f);
    }

    /**
     * Enable the image selector UI radio buttons.
     */
    void enableDelegateSelectionButtons() {
        cpuOnlyButton.setEnabled(true);
        allDelegatesButton.setEnabled(true);
    }

    /**
     * Reset the selected image view to the default image,
     * and enable portions of the inference UI accordingly.
     */
    void displayDefaultImage() {
        setInferenceUIEnabled(false);
        enableImageSelector();
        enableDelegateSelectionButtons();
        clearPredictionResults();
        selectedImageView.setImageResource(R.drawable.ic_launcher_background);
        selectedImage = null;
    }

    /**
     * Clear previous inference results from the UI.
     */
    void clearPredictionResults() {
        predictedClassesView.setText("");
        detInferenceTimeView.setText("-- ms");
        clsInferenceTimeView.setText("-- ms");
        predictionTimeView.setText("-- ms");
    }

    /**
     * Load an image for inference and update the UI accordingly.
     * The image will be loaded asynchronously to the main UI thread.
     *
     * @param imagePath Path to the image relative to the the `assets/images/` folder
     */
    void loadImageFromStringAsync(String imagePath) {
        setInferenceUIEnabled(false);
        // Exit the main UI thread and load the image in the background.
        backgroundTaskExecutor.execute(() -> {
            // Background task
            Bitmap scaledImage;
            try (InputStream inputImage = getAssets().open("images/" + imagePath)) {
                selectedImage = BitmapFactory.decodeStream(inputImage);
                scaledImage = ImageProcessing.resizeAndPadMaintainAspectRatio(selectedImage, selectedImageView.getWidth(), selectedImageView.getHeight(), 0xFF);
            } catch (IOException e) {
                throw new RuntimeException(e.getMessage());
            }

            mainLooperHandler.post(() -> {
                // In main UI thread
                selectedImageView.setImageBitmap(scaledImage);
                setInferenceUIEnabled(true);
            });
        });
    }

    /**
     * Load an image for inference and update the UI accordingly.
     * The image will be loaded asynchronously to the main UI thread.
     *
     * @param imageUri URI to the image.
     */
    void loadImageFromURIAsync(Uri imageUri) {
        setInferenceUIEnabled(false);
        // Exit the main UI thread and load the image in the background.
        backgroundTaskExecutor.execute(() -> {
            // Background task
            Bitmap scaledImage;
            try {
                if (android.os.Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
                    selectedImage = ImageDecoder.decodeBitmap(ImageDecoder.createSource(getContentResolver(), imageUri), (decoder, info, src) -> {
                        decoder.setMutableRequired(true);
                    });
                } else {
                    selectedImage = MediaStore.Images.Media.getBitmap(getContentResolver(), imageUri);
                }
                scaledImage = ImageProcessing.resizeAndPadMaintainAspectRatio(selectedImage, selectedImageView.getWidth(), selectedImageView.getHeight(), 0xFF);
            } catch (IOException e) {
                throw new RuntimeException(e.getMessage());
            }

            mainLooperHandler.post(() -> {
                // In main UI thread
                selectedImageView.setImageBitmap(scaledImage);
                setInferenceUIEnabled(true);
            });
        });
    }

    /**
     * Run the classifier on the currently selected image.
     * Prediction will run asynchronously to the main UI thread.
     * Disables inference UI before inference and re-enables it afterwards.
     */
    // TODO: add Detector
    void updatePredictionDataAsync() {
        setInferenceUIEnabled(false);
        predictedClassesView.setText("Inferencing...");

        ImageClassification imageClassification;
        ObjectDetection objectDetection;
        if (cpuOnlyInference) {
            imageClassification = cpuOnlyClassifier;
            objectDetection = cpuOnlyDetector;
        } else {
            imageClassification = defaultDelegateClassifier;
            objectDetection = defaultDelegateDetector;
        }

        // Exit the main UI thread and execute the model in the background.
        backgroundTaskExecutor.execute(() -> {
            /*
                TODO
                    - Input : selectedImage
                    - preprocessing before Detector
                        resized image (640*640) + normalized
                    - Output of Detector
                        [1, 6, 8400]
                    - postprocessing after Detector
                        bbox for resized image + Image with drawn boxes
                        -> 여기서 bbox들을 origin size에 맞춰서 후처리해줘야한다..
                        -> 이미지는 어떻게 하지?.... 분류 결과까지할거라면 나중에한꺼번에 그리는게 나을 것 같다.
                    - preprocessing before Classifier
                        box list -> 반복문으로 돌리면서
                        각 box + normalzied box -> cropped image (normalizing은 detection에서 사용하는거 참고)
                        resized image (480*480) + normalized
                    - Output of Classifier
                        [1, 976]
                    - postprocessing after Classifier
                        convert to float
                        softmax 적용
                        top-k filtering
             */

            List<Rect2d> boxes = objectDetection.detectObjectsFromImage(selectedImage);
            long det_inferenceTime = objectDetection.getLastInferenceTime();
            long det_predictionTime = objectDetection.getLastPostprocessingTime() + det_inferenceTime + objectDetection.getLastPreprocessingTime();
            String det_inferenceTimeText = timeFormatter.format((double) det_inferenceTime / 1000000);
            String det_predictionTimeText = timeFormatter.format((double) det_predictionTime / 1000000);

            /* TODO
                Postprocess for Detection output
                    - as is
                        return pair of processed image, predicted labels
                        - [1, 6, 8400]의 모델 출력에서 각 bbox정보 원복 및 저장
                        - NMS
                        - 이미지에 box 그리기 + string return
                    - to be
                        bbox가 나오도록 변경 (bbox)
                        - 원본 이미지 활용 -> box정보 원복
                        - List에 box정보 추가
                        - NMS
                        - return list of target bboxes
                -> 후처리 함수는 box 그리는 부분만 빼면 될 것 같다.
            */

            // Background task
            Pair<Bitmap, ArrayList<String>> cls_result = imageClassification.predictClassesFromImage(selectedImage, boxes);
            Bitmap cls_resultImage = cls_result.first;
            ArrayList<String> cls_resultLabels = cls_result.second;
            String cls_labels = String.join("\n", cls_resultLabels);
            double cls_totalInferenceTime = imageClassification.getTotalInfTime();
            double cls_averageInferenceTime = imageClassification.getAverageInferenceTime();

            Log.d("debugInferenceTime", "cls total inference time: " + cls_totalInferenceTime);
            Log.d("debugInferenceTime", "cls average inference time : " + cls_averageInferenceTime);

            /* TODO
                각 region별 inference 시간이 infTimeList에 담겨있고..
                여기서 1장당 평균 추론시간해서 Cls Inference Time에 표시하고..
                Postprocess시간도, preprocess시간도 그럼 각각 다 받아서.. 정확하게 하자
                cls_inference는 평균시간으로 보여주고..
                End-to-End시간은 전처리+후처리를 포함한 detection + 전처리+후처리를 포함한 모든 region classification까지
                그럼 total cls-inference시간을 하나 더 선언해야할 것
             */


//            double cls_predictionTime = imageClassification.getLastPostprocessingTime() + cls_totalInferenceTime + imageClassification.getLastPreprocessingTime();
            double cls_predictionTime = imageClassification.getTotalPreprocessTime() + cls_totalInferenceTime + imageClassification.getTotalPostprocesstime();

            String cls_inferenceTimeText = timeFormatter.format(cls_averageInferenceTime);
            String cls_predictionTimeText = timeFormatter.format(cls_predictionTime);

            String totalPredictionTimeText = timeFormatter.format((double) det_predictionTime/1000000 + cls_predictionTime);

            /*
            TODO: set ui value, TimeView 추가
                - 위에서 Detector 후처리 부분에서 변경사항도 존재..
                - Time도 일단 맨 처음과 맨마지막으로 해두자.
                이미지 변경완료했고.. 보여주는 디테일한 시간을 어플상으로 보이게 하자

            */
            mainLooperHandler.post(() -> {
                selectedImageView.setImageBitmap(cls_resultImage);
                // In main UI thread
                predictedClassesView.setText(cls_labels);
                detInferenceTimeView.setText(det_inferenceTimeText + " ms");
                clsInferenceTimeView.setText(cls_inferenceTimeText + " ms");
                predictionTimeView.setText(totalPredictionTimeText + " ms");
                setInferenceUIEnabled(true);
            });

            Log.d("check", "check");

        });
    }

    /**
     * Create inference classifier objects.
     * Loading the TF Lite model takes time, so this is done asynchronously to the main UI thread.
     * Disables the inference UI during load and reenables it afterwards.
     */
    void createTFLiteModelAsync() {
        if (defaultDelegateClassifier != null || cpuOnlyClassifier != null) {
            throw new RuntimeException("Classifiers were already created");
        }
        if (defaultDelegateDetector != null || cpuOnlyDetector != null) {
            throw new RuntimeException("Detectors were already created");
        }
        setInferenceUIEnabled(false);

        // Exit the UI thread and instantiate the model in the background.
        backgroundTaskExecutor.execute(() -> {
            // Create two classifiers.
            // One uses the default set of delegates (can access NPU, GPU, CPU), and the other uses only XNNPack (CPU).
            String cls_tfLiteModelAsset = this.getResources().getString(R.string.cls_tfLiteModelAsset);
            String cls_tfLiteLabelsAsset = this.getResources().getString(R.string.cls_tfLiteLabelsAsset);

            String det_tfLiteModelAsset = this.getResources().getString(R.string.det_tfLiteModelAsset);
            String det_tfLiteLabelsAsset = this.getResources().getString(R.string.det_tfLiteLabelsAsset);


            try {
                // TODO: classifier
                long delegateClassifierInitTime;
                long delegateClassifierInitStartTime = System.nanoTime();
                defaultDelegateClassifier = new ImageClassification(
                        this,
                        cls_tfLiteModelAsset,
                        cls_tfLiteLabelsAsset,
                        ClsAIHubDefaults.delegatePriorityOrder /* AI Hub Defaults */
                );
                delegateClassifierInitTime = System.nanoTime() - delegateClassifierInitStartTime;
                Log.d("ModelInit", "Delegate Classifier Init Time: " + delegateClassifierInitTime / 1000000 + " ms");

                long cpuClassifierInitTime;
                long cpuClassifierInitStartTime = System.nanoTime();
                cpuOnlyClassifier = new ImageClassification(
                        this,
                        cls_tfLiteModelAsset,
                        cls_tfLiteLabelsAsset,
                        ClsAIHubDefaults.delegatePriorityOrderForDelegates(new HashSet<>() /* No delegates; cpu only */)
                );
                cpuClassifierInitTime = System.nanoTime() - cpuClassifierInitStartTime;
                Log.d("ModelInit", "CPU Classifier Init Time: " + cpuClassifierInitTime / 1000000 + " ms");

                Log.d("ModelInit", "Default Classifier Delegates: " + Arrays.deepToString(ClsAIHubDefaults.delegatePriorityOrder));
                Log.d("ModelInit", "CPU Only Classifier Delegates: " + Arrays.deepToString(ClsAIHubDefaults.delegatePriorityOrderForDelegates(new HashSet<>())));


                //TODO: detector
                long delegateDetectorInitTime;
                long delegateDetectorInitStartTime = System.nanoTime();
                defaultDelegateDetector = new ObjectDetection(
                        this,
                        det_tfLiteModelAsset,
                        det_tfLiteLabelsAsset,
                        AIHubDefaults.delegatePriorityOrder
                );
                delegateDetectorInitTime = System.nanoTime() - delegateDetectorInitStartTime;
                Log.d("ModelInit", "Delegate Detector Init Time: " + delegateDetectorInitTime / 1000000 + " ms");

                long cpuDetectorInitTime;
                long cpuDetectorInitStartTime = System.nanoTime();
                cpuOnlyDetector = new ObjectDetection(
                        this,
                        det_tfLiteModelAsset,
                        det_tfLiteLabelsAsset,
                        AIHubDefaults.delegatePriorityOrderForDelegates(new HashSet<>() /* No delegates; cpu only */)
                );
                cpuDetectorInitTime = System.nanoTime() - cpuDetectorInitStartTime;
                Log.d("ModelInit", "CPU Detector Init Time: " + cpuDetectorInitTime / 1000000 + " ms");

                Log.d("ModelInit", "Default Detector Delegates: " + Arrays.deepToString(AIHubDefaults.delegatePriorityOrder));
                Log.d("ModelInit", "CPU Only Detector Delegates: " + Arrays.deepToString(AIHubDefaults.delegatePriorityOrderForDelegates(new HashSet<>())));

            } catch (IOException | NoSuchAlgorithmException e) {
                throw new RuntimeException(e.getMessage());
            }

            mainLooperHandler.post(() -> setInferenceUIEnabled(true));
        });
    }

    /**
     * Destroy this activity and release memory used by held objects.
     */
    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (cpuOnlyClassifier != null) cpuOnlyClassifier.close();
        if (defaultDelegateClassifier != null) defaultDelegateClassifier.close();

        // detector
        if (cpuOnlyDetector != null) cpuOnlyDetector.close();
        if (defaultDelegateDetector != null) defaultDelegateDetector.close();
    }
}
