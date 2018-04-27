package Analyzer;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;

import json.JSONArray;
import json.JSONException;
import json.JSONObject;
import structures._Doc4ETBIR;

/**
 * Created by lulin on 3/28/18.
 */
public class ReviewAnalyzer extends DocAnalyzer{

    protected String source;

    public ReviewAnalyzer(String tokenModel, int classNo, String providedCV, int Ngram, int threshold, String source)
                        throws IOException{
        super( tokenModel,  classNo,  providedCV,  Ngram,  threshold);
        this.source = source;
    }

    @Override
    public void LoadDoc(String filename) {
        if (filename.toLowerCase().endsWith(".json")) {
            if(source.equals("yelp"))
                LoadReviewYelp(filename);
            else
                LoadReviewAmazon(filename);
        }
        else
            System.out.println("!Wrong file suffix...");
    }

    public void LoadReviewYelp(String filename){
        JSONArray jarray = null;
        try{
            JSONObject json = LoadJSON(filename);
            jarray = json.getJSONArray("reviews");
        }catch (Exception e){
            System.err.print("!FAIL to parse a json file...");
            return;
        }

        JSONObject obj;
        _Doc4ETBIR review;
        String name, source, productID, userID, category = "";
        int ylabel;
        long timestamp = 0;
        for(int u = 0; u < jarray.length(); u++){
            try {
                obj = jarray.getJSONObject(u);
                name = obj.getString("review_id");
                source = obj.getString("text");
                userID = obj.getString("user_id");
                productID = obj.getString("business_id");
                ylabel = obj.getInt("stars");
                review = new _Doc4ETBIR(m_corpus.getSize(), name, productID, userID, source, ylabel, timestamp);
                AnalyzeDoc(review);
            }catch (JSONException e){
                System.out.println("!FAIL to parse a json object...");
            }
        }
    }

    //defined by Lu Lin for yelp review data
    public void LoadReviewAmazon(String filename){
        JSONArray jarray = null;
        try{
            JSONObject json = LoadJSON(filename);
            jarray = json.getJSONArray("reviews");
        }catch (Exception e){
            System.err.print("!FAIL to parse a json file...");
            return;
        }

        JSONObject obj;
        _Doc4ETBIR review;
        String name, text, productID, userID, category = "";
        int ylabel, reviewNum=0;
        long timestamp = 0;
        for(int u = 0; u < jarray.length(); u++){
            try {
                obj = jarray.getJSONObject(u);
                text = obj.getString("reviewText");
                userID = obj.getString("reviewerID");
                productID = obj.getString("asin");
                ylabel = obj.getInt("overall");
                review = new _Doc4ETBIR(m_corpus.getSize(), String.valueOf(reviewNum++), productID, userID, text, ylabel, timestamp);
                AnalyzeDoc(review);
            }catch (JSONException e){
                System.out.println("!FAIL to parse a json object...");
            }
        }
    }

}
