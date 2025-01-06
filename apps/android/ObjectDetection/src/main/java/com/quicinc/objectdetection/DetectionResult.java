package com.quicinc.objectdetection;

import android.graphics.Bitmap;

import java.util.List;


public class DetectionResult {
    public Bitmap annotatedImage; // bbox가 그려진 이미지
    public List<BboxInfo> bboxInfos; // bbox 정보 (인덱스, 클래스, confidence)

    public static class BboxInfo {
        public int index; // bbox 인덱스
        public String label; // 분류 결과
        public float confidence; // confidence score

        public BboxInfo(int index, String label, float confidence) {
            this.index = index;
            this.label = label;
            this.confidence = confidence;
        }
    }

    public DetectionResult(Bitmap annotatedImage, List<BboxInfo> bboxInfos) {
        this.annotatedImage = annotatedImage;
        this.bboxInfos = bboxInfos;
    }
}