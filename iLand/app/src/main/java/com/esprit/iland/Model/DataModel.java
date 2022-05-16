package com.esprit.iland.Model;

public class DataModel {
    private Boolean IsSelected;
    private Boolean IsValid;
    private Boolean IsCorrected;
    private String Handwritting;
    private String Chiffre;
    private String Message;


    public Boolean getSelected() {
        return IsSelected;
    }

    public void setSelected(Boolean selected) {
        IsSelected = selected;
    }

    public Boolean getValid() {
        return IsValid;
    }

    public void setValid(Boolean valid) {
        IsValid = valid;
    }


    public Boolean getCorrected() {
        return IsCorrected;
    }

    public void setCorrected(Boolean corrected) {
        IsCorrected = corrected;
    }

    public String getHandwritting() {
        return Handwritting;
    }

    public void setHandwritting(String handwritting) {
        Handwritting = handwritting;
    }

    public String getChiffre() {
        return Chiffre;
    }

    public void setChiffre(String chiffre) {
        Chiffre = chiffre;
    }

    public String getMessage() {
        return Message;
    }

    public void setMessage(String message) {
        Message = message;
    }
}
