package com.esprit.iland.Model;

import com.google.gson.annotations.SerializedName;

/**
 * Created by HoussemRIAHI on 5/4/2022.
 */

public class ResponseApiModel {
    @SerializedName("kode")
    String kode;
    @SerializedName("pesan")
    String pesan;

    public String getKode() {
        return kode;
    }

    public void setKode(String kode) {
        this.kode = kode;
    }

    public String getPesan() {
        return pesan;
    }

    public void setPesan(String pesan) {
        this.pesan = pesan;
    }
}
