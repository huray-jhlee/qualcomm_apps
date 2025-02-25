package com.quicinc.objectdetection;

import android.content.Context;
import android.util.Log;
import android.util.Pair;

import org.json.JSONArray;
import org.json.JSONObject;
import org.opencv.core.Rect2d;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class JsonFileUtils {
    private static final String TAG = "JsonFileUtils";

    /**
     * JSON 파일을 내부 저장소에 저장
     *
     * @param context  앱의 context
     * @param fileName 저장할 파일명 (예: "detection_results.json")
     * @param imagePath  이미지 파일 경로 (예: "/data2/jh/ondevice_test_images/images/00_resized.jpeg")
     * @param results 결과 리스트 (top1~top5) -> Pair<List<Rect2d>, ArrayList<String>>
     */
    public static void saveJsonToFile(Context context, String fileName, String imagePath, Pair<List<Rect2d>, ArrayList<String>> results, boolean isFirstRun) {
        try {
            // 내부 저장소 load : /data/data/com.quinic.xxx
            File directory = context.getFilesDir();
            File file = new File(directory, fileName);

            JSONObject jsonObject;
            Log.d("debug", "check");
            if (isFirstRun){
                jsonObject = new JSONObject();
            } else {
                if (file.exists()) {
                    jsonObject = new JSONObject(readFile(file));
                } else {
                    jsonObject = new JSONObject();
                }
            }

            JSONArray detectionsArray = new JSONArray();

            List<Rect2d> boxList = results.first;
            ArrayList<String> labelList = results.second;

            for (int i=0; i<boxList.size(); i++) {
                Rect2d box = boxList.get(i);
                String rawLabel = labelList.get(i);

                String [] splitLabel = rawLabel.split(" / ");
                String label = splitLabel[0];
                double conf = Double.parseDouble(splitLabel[1]);

                JSONArray detectionArray = new JSONArray();
                detectionArray.put(label);
                detectionArray.put(conf);

                JSONArray bboxArray = new JSONArray();
                bboxArray.put(box.x);
                bboxArray.put(box.y);
                bboxArray.put(box.x + box.width);
                bboxArray.put(box.y + box.height);

                detectionArray.put(bboxArray);
                detectionsArray.put(detectionArray);
            }

            jsonObject.put(imagePath, detectionsArray);

            FileWriter writer = new FileWriter(file, false); // append=false : 덮어쓰기
            writer.write(jsonObject.toString(4));
            writer.flush();
            writer.close();

            Log.d(TAG, "JSON 파일 저장 완료: " + file.getAbsolutePath());
        } catch (IOException e) {
            Log.e(TAG, "JSON 저장 실패", e);
        } catch (Exception e) {
            Log.e(TAG, "JSON 처리 중 오류 발생", e);
        }
    }
    private static String readFile(File file) throws IOException {
        StringBuilder content = new StringBuilder();
        BufferedReader reader = new BufferedReader(new FileReader(file));
        String line;
        while ((line = reader.readLine()) != null) {
            content.append(line);
        }
        reader.close();
        return content.toString();
    }
}
