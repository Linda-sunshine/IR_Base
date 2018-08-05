package Application;


import Analyzer.MultiThreadedReviewAnalyzer;
import json.JSONArray;
import json.JSONObject;
import opennlp.tools.util.InvalidFormatException;
import structures.*;

import java.io.*;
import java.util.ArrayList;
import java.util.HashMap;

/***
 * @author Lu Lin
 * Automatically tag item
 */
public class ItemTagging extends MultiThreadedReviewAnalyzer{
    protected ArrayList<_Review> m_tagSet;//store all the tags, each tag is a sparse vector of word
    protected ArrayList<_Item> m_items;
    protected HashMap<String, Integer> m_itemIDIndex;
    protected double[][] m_topic_word_probability;
    int m_embed_dim;

    public ItemTagging(String tokenModel, int classNo,
                       String providedCV, int Ngram, int threshold, int numberOfCores,
                       boolean b, String source)
            throws InvalidFormatException, FileNotFoundException, IOException {
        super(tokenModel, classNo, providedCV, Ngram, threshold, numberOfCores, b, source);
    }

    public void constructTagSet(String itemFileName){
        m_tagSet = new ArrayList<>();
        try{
            BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(itemFileName), "UTF-8"));
            String line;
            int doc_ID = 0;
            while((line=reader.readLine()) != null){
                JSONObject obj = new JSONObject(line.toString());
                if(obj.has("business_id")){
                    //parse json file, to get: itemID, categories
                    String itemID = obj.getString("business_id");
                    JSONArray categoryArray = obj.getJSONArray("categories");
                    String originTag = "";
                    for(int i = 0; i < categoryArray.length();i++)
                        originTag = originTag + ", " + categoryArray.getString(i);
                    //create doc for each item's tag
                    _Review doc = new _Review(doc_ID, originTag, 0);
                    doc.setItemID(itemID);
                    //analyze and add tag into tagset
                    analyzeTag(doc, 0);
                }
            }
        }catch (Exception e){
            System.err.format("[Error]FAIL to load item info file: %s\n", itemFileName);
        }
    }

    protected boolean analyzeTag(_Review doc, int core) {
        TokenizeResult result = TokenizerNormalizeStemmer(doc.getSource(),core);// Three-step analysis.
        String[] tokens = result.getTokens();
        int y = doc.getYLabel();

        // Construct the sparse vector.
        HashMap<Integer, Double> spVct = constructSpVct(tokens, y, null);

        doc.createSpVct(spVct);
        doc.setStopwordProportion(result.getStopwordProportion());

        m_tagSet.add(doc);
        return true;
    }

    public void loadItemWeight(String weightFile){
        //load item weights
        m_itemIDIndex = new HashMap<>();
        m_items = new ArrayList<>();
        String[] tokens = weightFile.split("\\.|\\_");
        int dim = Integer.valueOf(tokens[tokens.length-2]);
        m_embed_dim = dim;
        try{
            BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(weightFile), "UTF-8"));
            String line, itemID;
            String[] eta;
            double[] weight = new double[dim];
            int itemIndex=0;
            if(weightFile.contains("postEta")){
                while ((line = reader.readLine()) != null) {
                    itemIndex = Integer.valueOf(line.split("\\s+")[1]); // read item index
                    itemID = line.split("\\s+")[2]; // read item ID
                    m_itemIDIndex.put(itemID, itemIndex);
                    if(itemIndex == m_items.size())
                        m_items.add(new _Item(itemID));
                    else
                        System.err.println("[Warning]Load item weights: item index and doc not match.");

                    // read the eta of each item
                    eta = reader.readLine().split("\\s+");
                    weight = new double[eta.length];
                    for (int i = 0; i < eta.length; i++) {
                        weight[i] = Double.valueOf(eta[i]);
                    }
                    m_items.get(m_itemIDIndex.get(itemIndex)).setItemWeights(weight);
                }
            } else{
                while ((line = reader.readLine()) != null) {
                    itemID = line.split("[\\(|\\)|\\s]+")[1]; // // read item ID (format: ID xxx(30 reviews))
                    m_itemIDIndex.put(itemID, itemIndex);
                    m_items.add(new _Item(itemID));
                    // read the eta of each item
                    // read the p value, dim * dim
                    for (int d = 0; d < dim; d++) {
                        String p = reader.readLine().split("[\\(|\\)]+")[1];// read weight (format: -- Topic 0(0.03468):	...)
                        weight[d] = Double.valueOf(p);
                    }
                    m_items.get(m_itemIDIndex.get(itemIndex)).setItemWeights(weight);
                    itemIndex++;
                }
            }
            reader.close();
            System.out.format("[Info]Finish loading %d items' weights!\n", m_items.size());
        }catch (Exception e){
            System.err.format("[Error]FAIL to load item info file: %s\n", weightFile);
        }

    }

    public void buildItemProfile(){
        //allocate review by item
        for(_Doc doc : m_corpus.getCollection()){
            if(doc.getType() == _Review.rType.TEST)
                continue;
            String itemID = doc.getItemID();
            m_items.get(m_itemIDIndex.get(itemID)).addOneReview((_Review)doc);
        }
        //compress sparse feature of reviews into one vector for each item
        for(_Item item : m_items){
            item.buildProfile("");
        }
    }

    public void loadETBIRModel(String betaFile){//call after loadItemWeight
        try{
            BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(betaFile), "UTF-8"));
            String line;
            while((line=reader.readLine()) != null){

            }
        }catch (Exception e){
            System.err.format("[Error]FAIL to load item info file: %s\n", betaFile);
        }
    }
}
