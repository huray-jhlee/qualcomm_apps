// ---------------------------------------------------------------------
// Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// ---------------------------------------------------------------------
package com.quicinc.objectdetection;

import android.app.Activity;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.graphics.ImageDecoder;
import android.graphics.Paint;
import android.graphics.Canvas;
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

import java.io.IOException;
import java.io.InputStream;
import java.security.NoSuchAlgorithmException;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.stream.Collectors;
import java.util.List;

import org.opencv.android.OpenCVLoader;
import org.opencv.core.Rect2d;


public class MainActivity extends AppCompatActivity {

    static {
        System.loadLibrary("opencv_java4");
    }

    // UI Elements
    RadioGroup delegateSelectionGroup;
    RadioButton allDelegatesButton;
    RadioButton cpuOnlyButton;
    ImageView selectedImageView;
    TextView predictedClassesView;
    TextView inferenceTimeView;
    TextView predictionTimeView;
    Spinner imageSelector;
    Button predictionButton;
    ActivityResultLauncher<Intent> selectImageResultLauncher;
    private final String fromGalleryImageSelectorOption = "From Gallery";
    private final String notSelectedImageSelectorOption = "Not Selected";
    private final String[] imageSelectorOptions =
            { notSelectedImageSelectorOption,
                    "Sample1.png",
                    "Sample2.png",
                    "Sample3.png",
                    "katsu.jpg",
                    fromGalleryImageSelectorOption};

    // Inference Elements
    Bitmap selectedImage = null; // Raw image, not resized
    private ObjectDetection defaultDelegateDetector;
    private ObjectDetection cpuOnlyDetector;
    private boolean cpuOnlyDetection = false;
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
        if (!OpenCVLoader.initDebug()) {
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
        inferenceTimeView = (TextView)findViewById(R.id.inferenceTimeResultText);
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
                if (!cpuOnlyDetection) {
                    this.cpuOnlyDetection = true;
                    clearPredictionResults();
                }
            } else if (checkedId == R.id.defaultDelegateRadio) {
                if (cpuOnlyDetection) {
                    this.cpuOnlyDetection = false;
                    clearPredictionResults();
                }
            } else {
                throw new RuntimeException("A radio button for selected runtime is not implemented");
            }
        });

        // Setup button callback
        predictionButton.setOnClickListener((view) -> updatePredictionDataAsync());

        // Exit the UI thread and instantiate the model in the background.
        createTFLiteDetectorsAsync();

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
            inferenceTimeView.setText("-- ms");
            predictionTimeView.setText("-- ms");
            predictionButton.setEnabled(false);
            predictionButton.setAlpha(0.5f);
            imageSelector.setEnabled(false);
            imageSelector.setAlpha(0.5f);
            cpuOnlyButton.setEnabled(false);
            allDelegatesButton.setEnabled(false);
        } else if (cpuOnlyDetector != null && defaultDelegateDetector != null && selectedImage != null) {
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
        inferenceTimeView.setText("-- ms");
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
    void updatePredictionDataAsync() {
        setInferenceUIEnabled(false);
        predictedClassesView.setText("Inferencing...");

        ObjectDetection objectDetection;
        if (cpuOnlyDetection) {
            objectDetection = cpuOnlyDetector;
        } else {
            objectDetection = defaultDelegateDetector;
        }

        // Exit the main UI thread and execute the model in the background.
        backgroundTaskExecutor.execute(() -> {
            // Background task
//            Pair<Bitmap, ArrayList<String>> result = objectDetection.predictClassesFromImage(selectedImage);
            Pair<List<Rect2d>, ArrayList<String>> result = objectDetection.detectObjectsFromImage(selectedImage);
            List<Rect2d> boxes = result.first;
            ArrayList<String> labels = result.second;

            long det_inferenceTime = objectDetection.getLastInferenceTime();
            long det_predictionTime = objectDetection.getLastPostprocessingTime() + det_inferenceTime + objectDetection.getLastPreprocessingTime();
            String det_inferenceTimeText = timeFormatter.format((double) det_inferenceTime / 1000000);
            String det_predictionTimeText = timeFormatter.format((double) det_predictionTime / 1000000);


            // 이제.... boxes와 입력 이미지로 박스를 그려야 한다.

            // make processedBitmap
            Paint paint = new Paint();
            paint.setColor(Color.RED);
            paint.setStyle(Paint.Style.STROKE);
            paint.setStrokeWidth(2);

            Paint textPaint = new Paint();
            textPaint.setColor(Color.WHITE);
            textPaint.setTextSize(30);

            Bitmap mutableImage = selectedImage.copy(Bitmap.Config.ARGB_8888, true);
            Canvas canvas = new Canvas(mutableImage);
            int imgWidth = selectedImage.getWidth();
            int imgHeight = selectedImage.getHeight();

            for (int i=0; i<boxes.size(); i++) {
                Rect2d box = boxes.get(i);
                String text = labels.get(i);

                canvas.drawRect(
                        (float) box.x,
                        (float) box.y,
                        (float) (box.x + box.width),
                        (float) (box.y + box.height),
                        paint
                );

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


            }

            Bitmap processedBitmap = mutableImage;

            // 기존에 사용하던 Result를 변경..
//            Bitmap processedBitmap = result.first;
//            ArrayList<String> labels = result.second;
//             labels를 개행문자로 연결
            String joinedLabels = String.join("\n", labels);

            long inferenceTime = objectDetection.getLastInferenceTime();
            long predictionTime = objectDetection.getLastPostprocessingTime() + inferenceTime + objectDetection.getLastPreprocessingTime();
            String inferenceTimeText = timeFormatter.format((double) inferenceTime / 1000000);
            String predictionTimeText = timeFormatter.format((double) predictionTime / 1000000);

            mainLooperHandler.post(() -> {
                // In main UI thread
                selectedImageView.setImageBitmap(processedBitmap);  // ImageView에 표시
                predictedClassesView.setText(joinedLabels);
                inferenceTimeView.setText(inferenceTimeText + " ms");
                predictionTimeView.setText(predictionTimeText + " ms");
                setInferenceUIEnabled(true);
            });
        });
    }

    /**
     * Create inference classifier objects.
     * Loading the TF Lite model takes time, so this is done asynchronously to the main UI thread.
     * Disables the inference UI during load and reenables it afterwards.
     */
    void createTFLiteDetectorsAsync() {
        if (defaultDelegateDetector != null || cpuOnlyDetector != null) {
            throw new RuntimeException("Detectors were already created");
        }
        setInferenceUIEnabled(false);

        // Exit the UI thread and instantiate the model in the background.
        backgroundTaskExecutor.execute(() -> {
            // Create two classifiers.
            // One uses the default set of delegates (can access NPU, GPU, CPU), and the other uses only XNNPack (CPU).
            String tfLiteModelAsset = this.getResources().getString(R.string.tfLiteModelAsset);
            String tfLiteLabelsAsset = this.getResources().getString(R.string.tfLiteLabelsAsset);
            try {
                long delegateDetectorInitTime;
                long delegateDetectorInitStartTime = System.nanoTime();
                defaultDelegateDetector = new ObjectDetection(
                        this,
                        tfLiteModelAsset,
                        tfLiteLabelsAsset,
                        AIHubDefaults.delegatePriorityOrder /* AI Hub Defaults */
                );
                delegateDetectorInitTime = System.nanoTime() - delegateDetectorInitStartTime;
                Log.d("ObjectDetection", "Delegate Detector Init Time: " + delegateDetectorInitTime / 1000000 + " ms");

                long cpuDetectorInitTime;
                long cpuDetectorInitStartTime = System.nanoTime();
                cpuOnlyDetector = new ObjectDetection(
                        this,
                        tfLiteModelAsset,
                        tfLiteLabelsAsset,
                        AIHubDefaults.delegatePriorityOrderForDelegates(new HashSet<>() /* No delegates; cpu only */)
                );
                cpuDetectorInitTime = System.nanoTime() - cpuDetectorInitStartTime;
                Log.d("ObjectDetection", "CPU Detector Init Time: " + cpuDetectorInitTime / 1000000 + " ms");

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
        if (cpuOnlyDetector != null) cpuOnlyDetector.close();
        if (defaultDelegateDetector != null) defaultDelegateDetector.close();
    }
}
