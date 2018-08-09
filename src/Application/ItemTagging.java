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
    protected HashMap<String, _Review> m_tagSet;//store all the tags, each tag is a sparse vector of word
    protected ArrayList<_Item> m_items;
    protected HashMap<String, Integer> m_itemIDIndex;
    protected double[][] m_topic_word_probability;
    protected HashMap<Integer, Double> m_idf;
    double m_avedl;
    int m_embed_dim;
    String m_mode;
    String m_model;
    int m_top_k;

    public ItemTagging(String tokenModel, int classNo,
                       String providedCV, int Ngram, int threshold, int numberOfCores,
                       boolean b, String source)
            throws InvalidFormatException, FileNotFoundException, IOException {
        super(tokenModel, classNo, providedCV, Ngram, threshold, numberOfCores, b, source);
        m_model = "";
        m_mode = "";
    }

    public void setMode(String mode){ this.m_mode = mode; }

    public void setModel(String model){ this.m_model = model; }

    public void setTopK(int k){ this.m_top_k = k; }

    public void constructTagSet(String itemFileName){
        m_tagSet = new HashMap<>();
        try{
            BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(itemFileName), "UTF-8"));
            String line;
            int doc_ID = 0;
            while((line=reader.readLine()) != null){
                JSONObject obj = new JSONObject(line.toString());
                if(obj.has("business_id")){
                    //parse json file, to get: itemID, categories
                    String itemID = obj.getString("business_id");
                    if(!m_itemIDIndex.containsKey(itemID))
                        continue;
                    JSONArray categoryArray = obj.getJSONArray("categories");
                    String originTag = "";
                    for(int i = 0; i < categoryArray.length();i++)
                        originTag = originTag + ", " + categoryArray.getString(i);
                    //create doc for each item's tag
                    _Review doc = new _Review(doc_ID, originTag, 0);
                    doc.setItemID(itemID);
                    //analyze and add tag into tagset
                    analyzeTag(doc, 0);
                    m_tagSet.put(itemID, doc);
                }
            }
            System.out.format("[Info]Load %d tags\n", m_tagSet.size());
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
                    m_items.get(m_itemIDIndex.get(itemID)).setItemWeights(weight);
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
                    m_items.get(m_itemIDIndex.get(itemID)).setItemWeights(weight);
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
        double lenTotal = 0;
        for(_Item item : m_items){
            item.buildProfile("");
            lenTotal += item.getLength();
        }
        m_avedl = lenTotal / m_items.size();
    }

    public void loadModel(String betaFile){//call after loadItemWeight, with m_embed_dim already loaded
        m_topic_word_probability = new double[m_embed_dim][m_featureNames.size()];
        try{
            BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(betaFile), "UTF-8"));
            String line;
            String[] word_prob;
            reader.readLine();//the first line is feature name, which is ordered as m_featureNameIndex
            int i=0;
            while((line=reader.readLine()) != null){
                word_prob = line.split("\\s+");
                if(word_prob.length == m_featureNames.size()){
                    for(int v=0; v < word_prob.length; v++)
                        m_topic_word_probability[i][v] = Double.valueOf(word_prob[v]);
                    i++;
                }else
                    System.err.format("[Error]Wrong dimension of feature in beta file: %s\n", betaFile);
            }
            if(i != m_embed_dim)
                System.err.format("[Error]Wrong dimension of embedding in beta file: %s\n", betaFile);
        }catch (IOException e){
            System.err.format("[Error]FAIL to load item info file: %s\n", betaFile);
        }
    }

    public void calcIDF(){
        m_idf = new HashMap<>();
        for(_Review doc : m_tagSet.values()){
            _SparseFeature[] fv = doc.getSparse();
            for(int i = 0; i < fv.length; i++){
                int idx = fv[i].getIndex();
                if(!m_idf.containsKey(idx))
                    m_idf.put(idx, 0.0);
            }
        }

        for(Integer idx : m_idf.keySet()) {
            for (_Item item : m_items) {
                if (item.getFeature().containsKey(idx))
                    m_idf.put(idx, m_idf.get(idx) + 1);
            }
            if(m_idf.get(idx) > 0)
                m_idf.put(idx, Math.log(m_items.size() / m_idf.get(idx)));
            else
                System.err.format("[Warning]This tag word does not exist in item's review: %s\n", m_featureNames.get(idx));
        }
    }

    public void calculateTagging(String rankfile){
        if(m_mode.equals("") || m_model.equals("")){
            System.err.format("[Error]Item tagging model or mode not initialized...");
            return;
        }

        double map = 0;
        int numValid = 0, numInvalid = 0;
        if(m_mode.equals("Embed")){
            for(_Item item : m_items){
                MyPriorityQueue<_RankItem> fVector = new MyPriorityQueue<_RankItem>(m_tagSet.size());
                for (String id : m_tagSet.keySet()) {
                    double loglikelihood = 0, v;
                    int wid;
                    _SparseFeature[] fv = m_tagSet.get(id).getSparse();
                    for(int n=0; n<fv.length; n++) {
                        wid = fv[n].getIndex();
                        v = fv[n].getValue();
                        double p_w_on_item = 0;
                        for(int i = 0; i < m_embed_dim; i++)
                            p_w_on_item += m_topic_word_probability[i][wid] * Math.pow(item.getItemWeights()[i], 2);
                        loglikelihood += v * Math.log(p_w_on_item);
                    }
                    int len = m_tagSet.get(id).getTotalDocLength();
                    loglikelihood /= len;
                    fVector.add(new _RankItem(id, loglikelihood));
                }

                //calculate MAP
                double rank = 0;
                for(int i = 0;i < m_top_k; i++){
                    if(fVector.get(i).m_name.equals(item.getID())){
                        rank = i+1;
                        break;
                    }
                }
                if(rank>0){
                    map += 1.0/rank;
                    numValid++;
                }else
                    numInvalid++;


                //print out ranking list
                File file = new File(String.format("%s%s-Embed-%d-%s.txt", rankfile, m_model, m_embed_dim, item.getID()));
                try{
                    file.getParentFile().mkdirs();
                    file.createNewFile();
                } catch(IOException e){
                    e.printStackTrace();
                }
                try {
                    PrintWriter rankWriter = new PrintWriter(file);
                    //first line is the true tag of this item: true, itemID, tag_lenght, word1_idx(word1),...
                    _SparseFeature[] fv  = m_tagSet.get(item.getID()).getSparse();
                    rankWriter.format("true, %s, %d, ", item.getID(), fv.length);
                    for(int n = 0; n < fv.length; n++){
                        rankWriter.format("%d(%s), ", fv[n].getIndex(), m_featureNames.get(fv[n].getIndex()));
                    }
                    rankWriter.println();

                    int i = 0;
                    for (_RankItem it : fVector){
                        fv  = m_tagSet.get(it.m_name).getSparse();
                        rankWriter.format("%d, %s, %d, ", i++, it.m_name, fv.length);
                        for(int n = 0; n < fv.length; n++){
                            rankWriter.format("%d(%s), ", fv[n].getIndex(), m_featureNames.get(fv[n].getIndex()));
                        }
                        rankWriter.format("%f\n", it.m_value);
                    }
                    rankWriter.close();
                }catch(Exception ex){
                    System.err.format("[Error]File %s Not Found",rankfile);
                }
            }
        }else if(m_mode.equals("BM25")){
            calcIDF();
            for(_Item item : m_items){
                MyPriorityQueue<_RankItem> fVector = new MyPriorityQueue<_RankItem>(m_tagSet.size());
                for (String id : m_tagSet.keySet()) {
                    double loglikelihood = 0, v;
                    int wid;
                    _SparseFeature[] fv = m_tagSet.get(id).getSparse();
                    for(int n=0; n<fv.length; n++) {
                        wid = fv[n].getIndex();
                        v = fv[n].getValue();
                        double tf = item.calcAdditiveSmoothedProb(wid);
                        double idf = m_idf.get(wid);
                        loglikelihood += v * idf * tf * (1.5+1) /
                                (tf + 1.5 * (1 - 0.75 + 0.75 * item.getLength() / m_avedl));
                    }
                    loglikelihood /= m_tagSet.get(id).getTotalDocLength();
                    fVector.add(new _RankItem(id, loglikelihood));
                }

                //calculate MAP
                double rank = 0;
                for(int i = 0;i < 10; i++){
                    if(fVector.get(i).m_name.equals(item.getID())){
                        rank = i+1;
                        break;
                    }
                }
                if(rank>0){
                    map += 1.0/rank;
                    numValid++;
                }else
                    numInvalid++;


                //print out ranking list
                File file = new File(String.format("%s%s-Language-%d-%s.txt", rankfile, m_model, m_embed_dim, item.getID()));
                try{
                    file.getParentFile().mkdirs();
                    file.createNewFile();
                } catch(IOException e){
                    e.printStackTrace();
                }
                try {
                    PrintWriter rankWriter = new PrintWriter(file);
                    //first line is the true tag of this item: true, itemID, tag_lenght, word1_idx(word1),...
                    _SparseFeature[] fv  = m_tagSet.get(item.getID()).getSparse();
                    rankWriter.format("true, %s, %d, ", item.getID(), fv.length);
                    for(int n = 0; n < fv.length; n++){
                        rankWriter.format("%d(%s), ", fv[n].getIndex(), m_featureNames.get(fv[n].getIndex()));
                    }
                    rankWriter.println();

                    int i = 0;
                    for (_RankItem it : fVector){
                        fv  = m_tagSet.get(it.m_name).getSparse();
                        rankWriter.format("%d, %s, %d, ", i++, it.m_name, fv.length);
                        for(int n = 0; n < fv.length; n++){
                            rankWriter.format("%d(%s), ", fv[n].getIndex(), m_featureNames.get(fv[n].getIndex()));
                        }
                        rankWriter.format("%f\n", it.m_value);
                    }
                    rankWriter.close();
                }catch(Exception ex){
                    System.err.format("[Error]File %s Not Found",rankfile);
                }
            }
        }else if(m_mode.equals("Language")){
            for(_Item item : m_items){
                MyPriorityQueue<_RankItem> fVector = new MyPriorityQueue<_RankItem>(m_tagSet.size());
                for (String id : m_tagSet.keySet()) {
                    double loglikelihood = 0, v;
                    int wid;
                    _SparseFeature[] fv = m_tagSet.get(id).getSparse();
                    for(int n=0; n<fv.length; n++) {
                        wid = fv[n].getIndex();
                        v = fv[n].getValue();
                        loglikelihood += v * Math.log(item.calcAdditiveSmoothedProb(wid));
                    }
                    loglikelihood /= m_tagSet.get(id).getTotalDocLength();
                    fVector.add(new _RankItem(id, loglikelihood));
                }

                //calculate MAP
                double rank = 0;
                for(int i = 0;i < 10; i++){
                    if(fVector.get(i).m_name.equals(item.getID())){
                        rank = i+1;
                        break;
                    }
                }
                if(rank>0){
                    map += 1.0/rank;
                    numValid++;
                }else
                    numInvalid++;


                //print out ranking list
                File file = new File(String.format("%s%s-Language-%d-%s.txt", rankfile, m_model, m_embed_dim, item.getID()));
                try{
                    file.getParentFile().mkdirs();
                    file.createNewFile();
                } catch(IOException e){
                    e.printStackTrace();
                }
                try {
                    PrintWriter rankWriter = new PrintWriter(file);
                    //first line is the true tag of this item: true, itemID, tag_lenght, word1_idx(word1),...
                    _SparseFeature[] fv  = m_tagSet.get(item.getID()).getSparse();
                    rankWriter.format("true, %s, %d, ", item.getID(), fv.length);
                    for(int n = 0; n < fv.length; n++){
                        rankWriter.format("%d(%s), ", fv[n].getIndex(), m_featureNames.get(fv[n].getIndex()));
                    }
                    rankWriter.println();

                    int i = 0;
                    for (_RankItem it : fVector){
                        fv  = m_tagSet.get(it.m_name).getSparse();
                        rankWriter.format("%d, %s, %d, ", i++, it.m_name, fv.length);
                        for(int n = 0; n < fv.length; n++){
                            rankWriter.format("%d(%s), ", fv[n].getIndex(), m_featureNames.get(fv[n].getIndex()));
                        }
                        rankWriter.format("%f\n", it.m_value);
                    }
                    rankWriter.close();
                }catch(Exception ex){
                    System.err.format("[Error]File %s Not Found",rankfile);
                }
            }

        }else{
            System.err.format("[Error]Mode %s for item tagging has not been developed\n", m_mode);
        }

        //printout results
        if(numValid>0)
            map /= numValid;
        System.out.format("[Stat]MAP for %s model, %s mode is: %.5f, with %d valid and %d invalid top %d tags\n", m_model, m_mode, map, numValid, numInvalid, m_top_k);
    }

}
