package com.esprit.iland;


import com.esprit.iland.Model.ResponseApiModel;

import okhttp3.MultipartBody;
import okhttp3.RequestBody;
import retrofit2.Call;
import retrofit2.http.Multipart;
import retrofit2.http.POST;
import retrofit2.http.Part;

/**
 * Created by HoussemRIAHI on 5/4/2022.
 */

public interface ApiServices {

    @Multipart
    @POST("uploadimage.php")
    Call<ResponseApiModel> uploadImage (@Part MultipartBody.Part image,@Part("nom") RequestBody Nom);


}
