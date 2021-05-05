package com.example.group7;

import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.Camera;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageCapture;
import androidx.camera.core.ImageCaptureException;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.lifecycle.LifecycleOwner;

import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import com.example.group7.ml.Model;
import com.google.common.util.concurrent.ListenableFuture;

import org.checkerframework.checker.nullness.qual.NonNull;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.Console;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.lang.reflect.Array;
import java.nio.ByteBuffer;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;

public class MainActivity extends AppCompatActivity {

    Bitmap bmp;
    ByteBuffer byteBuffer;
    private TextView tv;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        tv = (TextView) findViewById(R.id.textView);


        // Get file path from Intent and use it to retrieve Bitmap (image to analyze)
        Bundle extras = getIntent().getExtras();
        String filePath = extras.getString("path");
        File file = new File(filePath);
        bmp = BitmapFactory.decodeFile(file.getAbsolutePath());
        analyzeImage(findViewById(R.id.textView));
    }


    public void analyzeImage(View view) {
        bmp = Bitmap.createScaledBitmap(bmp, 224, 224, true);
        try {
            Model model = Model.newInstance(getApplicationContext());

            // Creates inputs for reference.
            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 224, 224, 3}, DataType.FLOAT32);

            TensorImage tensorImage = new TensorImage(DataType.FLOAT32);
            tensorImage.load(bmp);
            ByteBuffer byteBuffer = tensorImage.getBuffer();

            inputFeature0.loadBuffer(byteBuffer);

            // Runs model inference and gets result.
            Model.Outputs outputs = model.process(inputFeature0);
            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();
            float[] converted = outputFeature0.getFloatArray();
            String text = Arrays.toString(converted);
            int index = findLargestIndex(converted);
            String symbol = matchSymbol(index);

            // Releases model resources if no longer used.
            model.close();

            tv.setText(text + "\n" + "\n" + "Predicted: " + symbol);

        } catch (IOException e) {
            Log.e("MainActivity", "IOException");
        }
    }

    public int findLargestIndex(float[] array) {
        if (array == null) {
            return -1;
        }
        int max = 0;
        for (int i = 0; i< array.length; i++) {
            if (array[max] < array[i]) {
                max = i;
            }
        }
        return max;
    }

    public String matchSymbol(int index) {
        String[] symbolArray = new String[] {".","÷","8","=","5","4", "-","9","1","+","7","6","3", "×","2","0"};
        return symbolArray[index];
    }
}

