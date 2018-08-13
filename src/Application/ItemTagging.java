package Application;


import Analyzer.MultiThreadedReviewAnalyzer;
import cc.mallet.util.ArrayUtils;
import json.JSONArray;
import json.JSONObject;
import opennlp.tools.util.InvalidFormatException;
import structures.*;
import utils.Utils;

import java.io.*;
import java.util.*;

/***
 * @author Lu Lin
 * Automatically tag item
 */
public class ItemTagging extends MultiThreadedReviewAnalyzer{
    private ArrayList<_Item> m_items;
    private Set<Integer> m_validItemIndex;
    private HashMap<String, Integer> m_itemIDIndex;
    private double[][] m_topic_word_probability;
    private HashMap<Integer, Double> m_idf;
    private HashMap<Integer, Double> m_ref;
    private double m_avedl;
    private int m_embed_dim;
    private String m_mode;
    private String m_model;
    private int m_top_k;

    public ItemTagging(String tokenModel, int classNo,
                       String providedCV, int Ngram, int threshold, int numberOfCores,
                       boolean b, String source)
            throws InvalidFormatException, FileNotFoundException, IOException {
        super(tokenModel, classNo, providedCV, Ngram, threshold, numberOfCores, b, source);
        m_model = "";
        m_mode = "";
        m_itemIDIndex = new HashMap<>();
        m_items = new ArrayList<>();
        m_top_k = 10;
    }

    public void setMode(String mode){ this.m_mode = mode; }

    public void setModel(String model){ this.m_model = model; }

    public void setTopK(int k){ this.m_top_k = k; }

    //load a tag as a doc, with expanding vocabulary and index items
    public void loadCorpus(String itemFileName){
        try{
            BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(itemFileName), "UTF-8"));
            String line;
            HashMap<Set<Integer>, Integer> duplicate = new HashMap<>();
            while((line=reader.readLine()) != null){
                JSONObject obj = new JSONObject(line.toString());
                if(obj.has("business_id")){
                    //parse json file, to get: itemID, categories
                    String itemID = obj.getString("business_id");
                    if(!m_itemIDIndex.containsKey(itemID)) {
                        m_itemIDIndex.put(itemID, m_items.size());
                        m_items.add(new _Item(itemID));
                    }else{
                        System.err.format("[Error]Duplicate item in business.json\n");
                    }

                    JSONArray categoryArray = obj.getJSONArray("categories");
                    Set<Integer> curTag = new HashSet<>();
                    for(int i = 0; i < categoryArray.length();i++) {
                        TokenizeResult result = TokenizerNormalizeStemmer(categoryArray.getString(i),0);// Three-step analysis.
                        String[] tokens = result.getTokens();
                        // Construct the sparse vector.
                        HashMap<Integer, Double> spVct = constructSpVct(tokens, 0, null);
//                        for(String tk : tokens) {
//                            expandVocabulary(tk);
//                            int index = m_featureNameIndex.get(tk);
//                            spVct.put(index, 1.0);
//
//                        }
                        if(spVct.size()<=0)
                            continue;

                        Set<Integer> tagIdxSet = new HashSet<>();
                        for (Integer idx : spVct.keySet())
                            tagIdxSet.add(idx);

                        if( !duplicate.containsKey(tagIdxSet) ) {
                            _Review doc = new _Review(m_corpus.getSize(), categoryArray.getString(i), 0);
                            doc.createSpVct(spVct);
                            m_corpus.addDoc(doc);
                            duplicate.put(tagIdxSet, doc.getID());
                        }
                        curTag.add(duplicate.get(tagIdxSet));
                    }
                    m_items.get(m_itemIDIndex.get(itemID)).setTag(curTag);
                }
            }
            System.out.format("[Info]Load %d tags, with %d features.\n", m_corpus.getCollection().size(), m_featureNames.size());
        }catch (Exception e){
            System.err.format("[Error]FAIL to load item info file: %s\n", itemFileName);
        }
    }

    public void loadItemWeight(String weightFile, int mode){
        //load item weights
        m_validItemIndex = new HashSet<>();
        String[] tokens = weightFile.split("\\.|\\_");
        int dim = Integer.valueOf(tokens[tokens.length-2]);
        m_embed_dim = dim;
        try{
            BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(weightFile), "UTF-8"));
            String line, itemID;
            String[] eta;
            double[] weight = new double[dim];
            if(weightFile.contains("postEta")){
                while ((line = reader.readLine()) != null) {
                    itemID = line.split("\\s+")[2]; // read item ID
                    // read the eta of each item
                    eta = reader.readLine().split("\\s+");
                    weight = new double[eta.length];
                    for (int i = 0; i < eta.length; i++) {
                        weight[i] = Double.valueOf(eta[i]);
                    }
                    //only enable top-k topic
                    if(mode>0) {
                        Set<Integer> topIdx = new HashSet<>();
                        for (int k = 0; k < mode; k++) {
                            double max = 0;
                            int idx = 0;
                            for (int i = 0; i < weight.length; i++) {
                                if (weight[i] > max && !topIdx.contains(i)) {
                                    idx = i;
                                    max = weight[i];
                                }
                            }
                            topIdx.add(idx);
                        }
                        for (int i = 0; i < weight.length; i++) {
                            if (!topIdx.contains(i))
                                weight[i] = 0;
                        }
                    }
                    m_items.get(m_itemIDIndex.get(itemID)).setItemWeights(weight);
                    m_validItemIndex.add(m_itemIDIndex.get(itemID));
                }
            } else{
                while ((line = reader.readLine()) != null) {
                    itemID = line.split("[(|)|\\s]+")[1]; // // read item ID (format: ID xxx(30 reviews))
                    // read the eta of each item
                    // read the p value, dim * dim
                    for (int d = 0; d < dim; d++) {
                        String p = reader.readLine().split("[(|)]+")[1];// read weight (format: -- Topic 0(0.03468):	...)
                        weight[d] = Double.valueOf(p);
                    }
                    //only enable top-k topic
                    if(mode>0) {
                        Set<Integer> topIdx = new HashSet<>();
                        for (int k = 0; k < mode; k++) {
                            double max = 0;
                            int idx = 0;
                            for (int i = 0; i < weight.length; i++) {
                                if (weight[i] > max && !topIdx.contains(i)) {
                                    idx = i;
                                    max = weight[i];
                                }
                            }
                            topIdx.add(idx);
                        }
                        for (int i = 0; i < weight.length; i++) {
                            if (!topIdx.contains(i))
                                weight[i] = 0;
                        }
                    }
                    m_items.get(m_itemIDIndex.get(itemID)).setItemWeights(weight);
                    m_validItemIndex.add(m_itemIDIndex.get(itemID));
                }
            }
            reader.close();
            System.out.format("[Info]Finish loading %d items' weights in total %d items!\n", m_validItemIndex.size(), m_items.size());
        }catch (Exception e){
            System.err.format("[Error]FAIL to load item info file: %s\n", weightFile);
        }

    }

    //build query for one item from its' reviews
    public void buildItemProfile(ArrayList<_Doc> docs){
        //allocate review by item
        for(_Doc doc : docs){
            if(doc.getType() == _Review.rType.TEST)
                continue;
            String itemID = doc.getItemID();
            m_items.get(m_itemIDIndex.get(itemID)).addOneReview((_Review)doc);
        }
        //compress sparse feature of reviews into one vector for each item
        double lenTotal = 0;
        for(_Item item : m_items){
            item.buildProfile("");
        }
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

    //calc idf and average doc length for documents(tags)
    private void calcStat(){
        m_avedl = 0;
        m_idf = new HashMap<>();
        m_ref = new HashMap<>();
        for(_Doc doc : m_corpus.getCollection()){
            _SparseFeature[] fv = doc.getSparse();
            for(int i = 0; i < fv.length; i++){
                int idx = fv[i].getIndex();
                if(!m_idf.containsKey(idx))
                    m_idf.put(idx, 0.0);
                m_idf.put(idx, m_idf.get(idx) + 1);

                double val = fv[i].getValue();
                if(!m_ref.containsKey(idx))
                    m_ref.put(idx, 0.0);
                m_ref.put(idx, m_ref.get(idx) + val);
            }
            m_avedl += doc.getTotalDocLength();
        }

        for(Integer idx : m_idf.keySet()) {
            m_idf.put(idx, Math.log(m_corpus.getSize() / m_idf.get(idx)));
        }

        m_avedl /= m_corpus.getSize();
    }

    public double[] calculateTagging(String rankfile, double lambda){//lambda is for interpolation or smoothing
        System.out.format("[Info]Begin tagging %d items with %s %s............\n", m_validItemIndex.size(), m_model, m_mode);

        double map_ave = 0, precision_ave = 0, mrr_ave = 0;
        int progress=0, invalid = 0;
        for(Integer validIdx : m_validItemIndex) {
            _Item item = m_items.get(validIdx);
            MyPriorityQueue<_RankItem> fVector = new MyPriorityQueue<_RankItem>(m_corpus.getSize());

            for (_Doc doc : m_corpus.getCollection()) {//for each doc, aka tag, compute the likelihood
                double loglikelihood = 0;
                HashMap<Integer, Double> fv_doc = Utils.revertSpVct(doc.getSparse());

                if (m_mode.equals("BM25")) {
                    calcStat();
                    double idf = 0, tf = 0, qtf = 0, b = 1.0, k1 = 1.5, k2 = 500;
                    for (Map.Entry<Integer, Double> entry : item.getFeature().entrySet()) {//for each word in its' review, aka query
                        //calc idf
                        idf = m_idf.containsKey(entry.getKey()) ? m_idf.get(entry.getKey()) : 0;
                        //calc tf
                        tf = fv_doc.containsKey(entry.getKey()) ? fv_doc.get(entry.getKey()) / doc.getTotalDocLength() : 0;
                        //calc qtf from item's review
                        qtf = entry.getValue() / item.getLength();

                        loglikelihood += entry.getValue() * idf * tf * (k1 + 1) /
                                (tf + k1 * (1 - b + b * doc.getTotalDocLength() / m_avedl)) * qtf * (k2 + 1) / (k2 + qtf);
                    }
                } else if (m_mode.equals("LM")) {
                    calcStat();
                    double delta = 0.1;
                    for (Map.Entry<Integer, Double> entry : item.getFeature().entrySet()) {//for each word in review, aka query
                        double mle_prob = doc.getTotalDocLength() >= 0 && fv_doc.containsKey(entry.getKey()) ? fv_doc.get(entry.getKey()) / doc.getTotalDocLength() : 0;
                        double ref_prob = m_ref.containsKey(entry.getKey()) ? (m_ref.get(entry.getKey())+delta)/(m_avedl+delta*m_featureNames.size()): delta/(m_avedl+delta*m_featureNames.size());
                        double smooth_prob = (1 - lambda) * mle_prob + lambda * ref_prob;
                        loglikelihood += entry.getValue() * Math.log((1 - lambda) * mle_prob + lambda * ref_prob);
                    }
                } else if (m_mode.equals("Embed")) {
                    for (Map.Entry<Integer, Double> entry : fv_doc.entrySet()) {
                        double p_w_on_item = 0;
                        for (int i = 0; i < m_embed_dim; i++)
                            p_w_on_item += m_topic_word_probability[i][entry.getKey()] * item.getItemWeights()[i];
                        loglikelihood += entry.getValue() * Math.log(p_w_on_item);
                    }
                    loglikelihood /= doc.getTotalDocLength();
                } else if(m_mode.equals("Interpolation")){
                    double score1 = 0, score2=0;
                    //first calc BM25
                    calcStat();
                    double idf = 0, tf = 0, qtf = 0, b = 1.0, k1 = 1.5, k2 = 500;
                    for (Map.Entry<Integer, Double> entry : item.getFeature().entrySet()) {//for each word in its' review, aka query
                        //calc idf
                        idf = m_idf.containsKey(entry.getKey()) ? m_idf.get(entry.getKey()) : 0;
                        //calc tf
                        tf = fv_doc.containsKey(entry.getKey()) ? fv_doc.get(entry.getKey()) / doc.getTotalDocLength() : 0;
                        //calc qtf from item's review
                        qtf = entry.getValue() / item.getLength();

                        score1 += entry.getValue() * idf * tf * (k1 + 1) /
                                (tf + k1 * (1 - b + b * doc.getTotalDocLength() / m_avedl)) * qtf * (k2 + 1) / (k2 + qtf);
                    }
                    //then calc Embed
                    for (Map.Entry<Integer, Double> entry : fv_doc.entrySet()) {
                        double p_w_on_item = 0;
                        for (int i = 0; i < m_embed_dim; i++)
                            p_w_on_item += m_topic_word_probability[i][entry.getKey()] * item.getItemWeights()[i];
                        score2 += entry.getValue() * Math.log(p_w_on_item);
                    }
                    score2 /= doc.getTotalDocLength();
                    //interpolation
                    loglikelihood = lambda * score1 + (1-lambda) * score2;
                }
                else{
                    System.err.format("[Error]Mode %s for item tagging has not been developed\n", m_mode);
                }

                fVector.add(new _RankItem(doc.getID(), loglikelihood));
            }

            //calculate precision@k
            double map=0, rel=0, mrr=0, firstHit=0;
            double hit = 0, trueTopK=0, precisionK = 0;
            int i=0;
            for (i = 0; i < m_top_k; i++) {
                if (item.getTag().contains(fVector.get(i).m_index)) {
                    hit += 1;
                }
                if(hit == item.getTag().size()){
                    break;
                }
            }
            precisionK = hit/(i+1);
            precision_ave += precisionK;

            //calc map
            i = 0;
            hit=0;
            rel=0;
            for(_RankItem it : fVector){
                i++;
                if(item.getTag().contains(it.m_index)){
                    if(hit==0)
                        firstHit = i;
                    hit += 1;
                    rel += hit/i;
                }
                if(hit == item.getTag().size()){
                    break;
                }
            }
            map = item.getTag().size() > 0? rel/hit : 0;
            map_ave += map;
            mrr = item.getTag().size() > 0? 1/firstHit : 0;
            mrr_ave += mrr;

            if(progress++ % 50 == 0)
                System.out.format("---- item %d: map is %.5f, precision@k is %.5f, mrr is %.5f\n", progress++, map, precisionK, mrr);

            //print out ranking list
//            File file = new File(String.format("%s%s-%s-%d-%s.txt", rankfile, m_model, m_mode, m_embed_dim, item.getID()));
//            try {
//                file.getParentFile().mkdirs();
//                file.createNewFile();
//            } catch (IOException e) {
//                e.printStackTrace();
//            }
//            try {
//                PrintWriter rankWriter = new PrintWriter(file);
//                //first line is the true tag of this item: true, itemID, tag_lenght, word1_idx(word1),...
//                rankWriter.format("true, %s, %d, ", item.getID(), item.getTag().size());
//                for (Integer idx : item.getTag()) {
//                    rankWriter.format("%d(%s), ", idx, m_corpus.getCollection().get(idx).getSource());
//                }
//                rankWriter.println();
//
//                i = 0;
//                for (_RankItem it : fVector) {
//                    rankWriter.format("%d, %d, true: %s, %f, filtered: ", i++, it.m_index, m_corpus.getCollection().get(it.m_index).getSource(), it.m_value);
//                    _SparseFeature[] fv = m_corpus.getCollection().get(it.m_index).getSparse();
//                    for(int m = 0;m < fv.length; m++){
//                        rankWriter.format("%s, ", m_featureNames.get(fv[m].getIndex()));
//                    }
//                    rankWriter.println();
//                }
//                rankWriter.close();
//            } catch (Exception ex) {
//                System.err.format("[Error]File %s Not Found", rankfile);
//            }
        }

        map_ave /= m_validItemIndex.size() - invalid;
        precision_ave /= m_validItemIndex.size() - invalid;
        mrr_ave /= m_validItemIndex.size() - invalid;

        //printout results
        System.out.format("[Stat]%s model, %s mode item tagging: Precision@K is %.5f, MAP is: %.5f, MRR is: %.5f, in top %d tags\n", m_model, m_mode, precision_ave, map_ave, mrr_ave, m_top_k);

        double[] result = new double[3];
        result[0] = map_ave;
        result[1] = mrr_ave;
        result[2] = precision_ave;

        return result;
    }

}
