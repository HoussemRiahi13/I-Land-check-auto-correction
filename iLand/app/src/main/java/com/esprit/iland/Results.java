package com.esprit.iland;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;

import android.app.ProgressDialog;
import android.content.Context;
import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;

import com.esprit.iland.Model.DataModel;

import org.w3c.dom.Text;

import java.io.IOException;

import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;
import retrofit2.Retrofit;
import retrofit2.converter.gson.GsonConverterFactory;

public class Results extends AppCompatActivity {
    Button Retour;
    TextView IsSelected;
    TextView IsValid;
    TextView IsCorrected;
    TextView Handwriting;
    TextView Chiffre;
    ProgressDialog pd;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_results);
        Retour=findViewById(R.id.Retour);
        IsSelected=findViewById(R.id.isSelected);
        IsValid=findViewById(R.id.isValid);
        IsCorrected=findViewById(R.id.isCorrected);
        Handwriting=findViewById(R.id.Handwriting);
        Chiffre=findViewById(R.id.Chiffre);
        Results.this.runOnUiThread(new Runnable() {
            @Override
            public void run() {
                if(MainActivity.IsSelected){
                    IsSelected.setText("✅");
                }else{
                    IsSelected.setText("❌");
                }
                if(MainActivity.IsCorrected){
                    IsCorrected.setText("✅");
                }else{
                    IsCorrected.setText("❌");
                }
                if(MainActivity.IsValid){
                    IsValid.setText("✅");
                }else{
                    IsValid.setText("❌");
                }
                if(!MainActivity.HandWrinting.equals("")){
                    Handwriting.setText(MainActivity.HandWrinting);
                }else{
                    Handwriting.setText("❌");
                }
                if(!MainActivity.Chiffre.equals("")){
                    Chiffre.setText(MainActivity.Chiffre);
                }else{
                    Chiffre.setText("❌");
                }
                    //myResponse
            }
        });

        Retour.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent intent=new Intent(Results.this,MainActivity.class);
                startActivity(intent);

            }
        });
    }
}