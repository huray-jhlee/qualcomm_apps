package com.quicinc.imageclassification;

import android.content.Context;
import android.util.Log;
import android.util.Pair;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;

public class JsonFileUtils {
    private static final String TAG = "JsonFileUtils";

    /**
     * JSON 파일을 내부 저장소에 저장
     *
     * @param context  앱의 context
     * @param fileName 저장할 파일명 (예: "classification_results.json")
     * @param imagePath  이미지 파일 경로 (예: "/data2/jh/ondevice_test_images/images/00_resized.jpeg")
     * @param resultsList 결과 리스트 (top1~top5) -> ArrayList<Pair<String, Float>>
     */
    public static void saveJsonToFile(Context context, String fileName, String imagePath, ArrayList<Pair<String, Float>> resultsList, boolean isFirstRun) {
        try {
            // 내부 저장소 load : /data/data/com.quinic.xxx
            File directory = context.getFilesDir();
            File file = new File(directory, fileName);

            JSONObject jsonObject;

            if (isFirstRun){
                jsonObject = new JSONObject();
            } else {
                if (file.exists()) {
                    jsonObject = new JSONObject(readFile(file));
                } else {
                    jsonObject = new JSONObject();
                }
            }

            JSONArray predictionsArray = new JSONArray();
            for (Pair<String, Float> result: resultsList) {
                predictionsArray.put(result.first + "-" + String.format("%.5f", result.second));
            }

            jsonObject.put(imagePath, predictionsArray);

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
