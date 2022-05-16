package com.esprit.iland;

import retrofit2.Retrofit;
import retrofit2.converter.gson.GsonConverterFactory;

/**
 * Created by HoussemRIAHI on 5/4/2022.
 */

public class RetroClient {
    private static  Retrofit retro = null;
    private static final String base_url = "https://8557-102-157-248-175.eu.ngrok.io/uploadimage/" ;


    private static Retrofit getClient()
    {
        if(retro == null)
        {
            retro  = new Retrofit.Builder()
                    .baseUrl(base_url)
                    .addConverterFactory(GsonConverterFactory.create())
                    .build();
        }

        return  retro;
    }

    public static ApiServices getApiServices()
    {
        return  getClient().create(ApiServices.class);
    }
}
