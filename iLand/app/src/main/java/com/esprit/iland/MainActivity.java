package com.esprit.iland;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.ContextCompat;

import android.Manifest;
import android.annotation.SuppressLint;
import android.app.ProgressDialog;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.RadioButton;
import android.widget.RadioGroup;
import android.widget.Switch;
import android.widget.Toast;

import com.esprit.iland.Model.DataModel;
import com.esprit.iland.Model.ResponseApiModel;

import org.apache.commons.lang3.RandomStringUtils;

import java.io.File;
import java.util.Objects;
import java.util.concurrent.TimeUnit;

import okhttp3.MediaType;
import okhttp3.MultipartBody;
import okhttp3.OkHttpClient;
import okhttp3.RequestBody;
import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;
import retrofit2.Retrofit;
import retrofit2.converter.gson.GsonConverterFactory;

public class MainActivity extends AppCompatActivity {

    Button btnUpload, btnGalery;
    ImageView imgHolder;
    String part_image;
    ProgressDialog pd;
    RadioButton Ar;
    RadioButton Fr;
    RadioButton Eng;
    final int REQUEST_GALLERY = 9544;
    public static String type;
    public static String Nom;
    public static Boolean IsSelected;
    public static Boolean IsValid;
    public static Boolean IsCorrected;
    public static String HandWrinting;
    public static String Chiffre;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);


        btnUpload = (Button) findViewById(R.id.btnupload);
        btnGalery= (Button) findViewById(R.id.btngallery);
        imgHolder = (ImageView) findViewById(R.id.imgHolder);
        Ar=(RadioButton)findViewById(R.id.ar) ;
        Fr=(RadioButton)findViewById(R.id.fr) ;
        Eng=(RadioButton)findViewById(R.id.eng) ;
        pd = new ProgressDialog(this);
        pd.setMessage("loading ... ");

        btnGalery.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent();
                intent.setType("image/*");
                intent.setAction(Intent.ACTION_GET_CONTENT);
                startActivityForResult(Intent.createChooser(intent,"open gallery"),REQUEST_GALLERY);
            }
        });


        String result = RandomStringUtils.random(12, true, true);
        Nom=result;
        btnUpload.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                pd.show();
                File imagefile = new File(part_image);
                RequestBody reqBody = RequestBody.create(MediaType.parse("multipart/form-file"),imagefile);
                MultipartBody.Part partImage = MultipartBody.Part.createFormData("imageupload", imagefile.getName(),reqBody);
                RequestBody reqNom=RequestBody.create(MediaType.parse("multipart/form-data"),result);

                ApiServices api = RetroClient.getApiServices();
                Call<ResponseApiModel> upload = api.uploadImage(partImage,reqNom);
                upload.enqueue(new Callback<ResponseApiModel>() {
                    @Override
                    public void onResponse(Call<ResponseApiModel> call, Response<ResponseApiModel> response) {
                        if(Ar.isChecked()){
                            type="True";
                        }else if(Fr.isChecked()){
                            type="False";
                        }else if(Ar.isChecked()){
                            type="eng";
                        }
                        OkHttpClient client = new OkHttpClient.Builder()
                                .connectTimeout(300, TimeUnit.SECONDS)
                                .readTimeout(300,TimeUnit.SECONDS).build();
                        Retrofit retrofit=new Retrofit.Builder().baseUrl("http://192.168.1.13:9595/").client(client).addConverterFactory(GsonConverterFactory.create()).build();
                        GetApiService getApiService=retrofit.create(GetApiService.class);
                        Call<DataModel> call1=getApiService.getData(Nom,type);
                        call1.enqueue(new Callback<DataModel>() {
                            @Override
                            public void onResponse(Call<DataModel> call, Response<DataModel> response) {

                                IsSelected = response.body().getSelected() ;
                                IsValid = response.body().getValid();
                                IsCorrected =response.body().getCorrected();
                                HandWrinting = response.body().getHandwritting();
                                Chiffre =response.body().getChiffre();
                                Intent intent=new Intent(MainActivity.this,Results.class);
                                startActivity(intent);
                                pd.dismiss();

                            }

                            @Override
                            public void onFailure(Call<DataModel> call, Throwable t) {
                                System.out.println("testttttttttt");
                            }
                        });


                    }

                    @Override
                    public void onFailure(Call<ResponseApiModel> call, Throwable t) {
                        if(Ar.isChecked()){
                            type="True";
                        }else if(Fr.isChecked()){
                            type="False";
                        }else if(Ar.isChecked()){
                            type="eng";
                        }
                        OkHttpClient client = new OkHttpClient.Builder()
                                .connectTimeout(300, TimeUnit.SECONDS)
                                .readTimeout(300,TimeUnit.SECONDS).build();
                        Retrofit retrofit=new Retrofit.Builder().baseUrl("http://192.168.1.13:9595/").client(client).addConverterFactory(GsonConverterFactory.create()).build();
                        GetApiService getApiService=retrofit.create(GetApiService.class);
                        Call<DataModel> call1=getApiService.getData(Nom,type);
                        call1.enqueue(new Callback<DataModel>() {
                            @Override
                            public void onResponse(Call<DataModel> call, Response<DataModel> response) {
                                try {
                                    IsSelected = Objects.requireNonNull(response.body()).getSelected();
                                }catch(NullPointerException e){
                                    e.printStackTrace();
                                }
                                try {
                                    IsValid = Objects.requireNonNull(response.body()).getValid();
                                }catch(NullPointerException e){
                                    e.printStackTrace();
                                }try{
                                    IsCorrected = Objects.requireNonNull(response.body()).getCorrected();
                                }catch(NullPointerException e){
                                    e.printStackTrace();
                                }
                                HandWrinting = response.body() != null ? response.body().getHandwritting() : null;
                                Chiffre =response.body() !=null ? response.body().getChiffre() : null;
                                Intent intent=new Intent(MainActivity.this,Results.class);
                                startActivity(intent);
                                pd.dismiss();

                            }

                            @Override
                            public void onFailure(Call<DataModel> call, Throwable t) {
                                System.out.println("testttttttttt");
                            }
                        });



                    }

                });

            }

        });

    }


    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        System.out.println("wsselt ");
        if (resultCode == RESULT_OK)
        {
            System.out.println("wsselt1 ");
            if(requestCode == REQUEST_GALLERY)
            {
                System.out.println("wsselt2 ");
                Uri dataimage = data.getData();
                String[] imageprojection = {MediaStore.Images.Media.DATA};
                Cursor cursor = getContentResolver().query(dataimage,imageprojection,null,null,null);

                if (cursor != null)
                {
                    System.out.println("wsselt3 ");
                    cursor.moveToFirst();
                    int indexImage = cursor.getColumnIndex(imageprojection[0]);
                    part_image = cursor.getString(indexImage);
                    System.out.println(part_image+"*********");

                    if(part_image != null)
                    {
                        if (ContextCompat.checkSelfPermission(
                                this,Manifest.permission.READ_EXTERNAL_STORAGE) ==
                                PackageManager.PERMISSION_GRANTED) {
                            // You can use the API that requires the permission.
                            System.out.println("wsselit 4");
                            File image = new File(part_image);
                            Bitmap myBitmap = BitmapFactory.decodeFile(image.getAbsolutePath());
                            imgHolder.setImageBitmap(myBitmap);}
                    } else {
                        // You can directly ask for the permission.
                        // The registered ActivityResultCallback gets the result of this request.
                        requestPermissionLauncher.launch(
                                Manifest.permission.READ_EXTERNAL_STORAGE);
                        if (ContextCompat.checkSelfPermission(
                                this,Manifest.permission.READ_EXTERNAL_STORAGE) ==
                                PackageManager.PERMISSION_GRANTED) {
                            // You can use the API that requires the permission.
                        }
                    }
                }

            }
        }
    }
    private ActivityResultLauncher<String> requestPermissionLauncher =
            registerForActivityResult(new ActivityResultContracts.RequestPermission(), isGranted -> {
                if (isGranted) {
                    System.out.println("wsselit 4");
                    File image = new File(part_image);
                    Bitmap myBitmap = BitmapFactory.decodeFile(image.getAbsolutePath());
                    imgHolder.setImageBitmap(myBitmap);
                } else {
                    // Explain to the user that the feature is unavailable because the
                    // features requires a permission that the user has denied. At the
                    // same time, respect the user's decision. Don't link to system
                    // settings in an effort to convince the user to change their
                    // decision.
                }
            });


}

