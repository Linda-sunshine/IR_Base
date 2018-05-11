package Analyzer;

import java.io.IOException;

import json.JSONArray;
import json.JSONException;
import json.JSONObject;
import structures._Doc4ETBIR;

/**
 * Created by lulin on 3/28/18.
 */
public class ReviewAnalyzer extends DocAnalyzer{

    public ReviewAnalyzer(String tokenModel, int classNo, String providedCV, int Ngram, int threshold)
                        throws IOException{
        super( tokenModel,  classNo,  providedCV,  Ngram,  threshold);
    }

    @Override
    public void LoadDoc(String filename) {
        if (filename.toLowerCase().endsWith(".json"))
            LoadReviewYelp(filename);
        else
            System.out.println("!Wrong file suffix...");
    }

    //defined by Lu Lin for yelp review data
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

}
