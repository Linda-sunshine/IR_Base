package Analyzer;

import java.io.IOException;

import json.JSONArray;
import json.JSONException;
import json.JSONObject;
import structures._Doc4ETBIR;

/**
 * Created by lulin on 3/28/18.
 */
public class ReviewAnalyzer extends UserAnalyzer{

    protected String source;

    public ReviewAnalyzer(String tokenModel, int classNo, String providedCV, int Ngram, int threshold, 
    		boolean b, String source) throws IOException{
        super(tokenModel,  classNo,  providedCV,  Ngram,  threshold, b);
        this.source = source;
    }

    @Override
    public void loadUser(String filename) {
        if (!filename.toLowerCase().endsWith(".json")) 
            System.err.println("[Error] Wrong file suffix...");

        String[] keys;
        if(source.equals("yelp"))
        	keys = new String[]{"review_id", "text", "user_id", "business_id", "stars"};
        else 
        	keys = new String[]{"reviewText", "reviewerID", "asin", "overall"};

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
        String name, text, productID, userID;
        int ylabel, reviewNum = 0;
        long timestamp = 0;
        for(int u = 0; u < jarray.length(); u++){
            try {
            	int index = 0;
                obj = jarray.getJSONObject(u);
                name = source.equals("yelp") ? obj.getString(keys[index++]) : String.valueOf(reviewNum++);
                text = obj.getString(keys[index++]);
                userID = obj.getString(keys[index++]);
                productID = obj.getString(keys[index++]);
                ylabel = obj.getInt(keys[index]);
                review = new _Doc4ETBIR(m_corpus.getSize(), name, productID, userID, text, ylabel, timestamp);
                AnalyzeDoc(review);
            }catch (JSONException e){
                System.out.println("!FAIL to parse a json object...");
            }
        }
    }
}
