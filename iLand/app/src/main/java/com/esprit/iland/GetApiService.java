package com.esprit.iland;

import com.esprit.iland.Model.DataModel;

import java.util.List;

import retrofit2.Call;
import retrofit2.http.GET;
import retrofit2.http.Query;

public interface GetApiService {


    @GET("Pi_DS/hello/")
    Call<DataModel> getData(@Query("nom") String nom,@Query("type") String type);
}
