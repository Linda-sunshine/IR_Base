package mains;

import Analyzer.BipartiteAnalyzer;
import Analyzer.MultiThreadedReviewAnalyzer;
import structures.*;

import java.io.*;
import java.text.ParseException;
import java.util.*;

public class myDataProcessMain {
    static List<_User> m_users;
    static List<_Item> m_items;

    static HashMap<String, Integer> m_usersIndex; //(userID, index in m_users)
    static HashMap<String, Integer> m_itemsIndex; //(itemID, index in m_items)
    static HashMap<String, Integer> m_reviewIndex; //(itemIndex_userIndex, index in m_corpus.m_collection)

    static BipartiteAnalyzer m_bipartite;
    static HashMap<Integer, ArrayList<Integer>> m_mapByUser;
    static HashMap<Integer, ArrayList<Integer>> m_mapByItem;

    public static void main(String[] args) throws IOException, ParseException {
        TopicModelParameter param = new TopicModelParameter(args);

        int classNumber = 6; //Define the number of classes in this Naive Bayes.
        int Ngram = 2; //The default value is unigram.
        String featureValue = "TF"; //The way of calculating the feature value, which can also be "TFIDF", "BM25"
        int norm = 0;//The way of normalization.(only 1 and 2)
        int lengthThreshold = 5; //Document length threshold
        int numberOfCores = Runtime.getRuntime().availableProcessors();
        String tokenModel = "./data/Model/en-token.bin";
        int crossV = 5;
        boolean flag_coldstart = true;

        /*****data setting*****/
        String folder = String.format("%s/%s/%s", param.m_prefix, param.m_source, param.m_set);
        String fvFile = String.format("%s/%s/%s_features.txt", param.m_prefix, param.m_source, param.m_source);
        String outputFolder = String.format("%s/%s/", folder, param.m_topicmodel);
        new File(outputFolder).mkdirs();

        String reviewFolder =  String.format("%s/%dfoldsCV%s/", folder, crossV, flag_coldstart?"Coldstart":"");
        MultiThreadedReviewAnalyzer analyzer = new MultiThreadedReviewAnalyzer(tokenModel, classNumber, fvFile,
                Ngram, lengthThreshold, numberOfCores, true, param.m_source);

        for(int k = 0; k <crossV; k++){
            //load test set
            String testFolder = reviewFolder + k + "/";
            analyzer.loadUserDir(testFolder);
            for(_Doc d : analyzer.getCorpus().getCollection()){
                d.setType(_Review.rType.TEST);
            }
            //load validation set
            int val=0;
            if(k < crossV-1)
                val = k + 1;
            String validationFolder = reviewFolder + val + "/";
            analyzer.loadUserDir(validationFolder);
            for(_Doc d : analyzer.getCorpus().getCollection()){
                if(d.getType()!=_Review.rType.TEST)
                    d.setType(_Review.rType.ADAPTATION);
            }
            //load train set
            for(int i = 0; i < crossV; i++){
                if(i!=k && i!=val){
                    String trainFolder = reviewFolder + i + "/";
                    analyzer.loadUserDir(trainFolder);
                }
            }

            m_bipartite = new BipartiteAnalyzer(analyzer.getCorpus());
            m_bipartite.analyzeCorpus();
            m_users = m_bipartite.getUsers();
            m_items = new ArrayList<>();
            for(_Product prd : m_bipartite.getItems()){
                m_items.add(new _Item(prd.getID()));
            }
            m_usersIndex = m_bipartite.getUsersIndex();
            m_itemsIndex = m_bipartite.getItemsIndex();
            m_reviewIndex = m_bipartite.getReviewIndex();

            ArrayList<_Doc> m_trainSet = new ArrayList<_Doc>();
            ArrayList<_Doc> m_testSet = new ArrayList<_Doc>();
            ArrayList<_Doc> m_validationSet = new ArrayList<>();
            for(_Doc d:analyzer.getCorpus().getCollection()){
                if(d.getType() == _Doc.rType.TRAIN){
                    m_trainSet.add(d);
                }else if(d.getType() == _Doc.rType.TEST){
                    m_testSet.add(d);
                }else if(d.getType() == _Doc.rType.ADAPTATION){
                    m_validationSet.add(d);
                }
            }

            String[] modes = new String[]{"train","validation","test"};
            ArrayList<_Doc> docs;
            for(String mode : modes) {
                if(mode.equals("train"))
                    docs = m_trainSet;
                else if(mode.equals("test"))
                    docs = m_testSet;
                else
                    docs = m_validationSet;
                m_bipartite.analyzeBipartite(docs, mode);
                m_mapByUser = m_bipartite.getMapByUser();
                m_mapByItem = m_bipartite.getMapByItem();

                if(param.m_topicmodel.equals("CTR")) {
                    save2FileCTPE(outputFolder, param.m_source, mode, k);
                }else{
                    save2FileRTM(outputFolder, param.m_source, mode, k);
                }
            }
        }
    }


    public static void save2FileCTPE(String prefix, String source, String mode, int k) throws IOException{
        ArrayList<_Doc> docs = new ArrayList<>();
        for(Map.Entry<Integer, ArrayList<Integer>> entry : m_mapByUser.entrySet()){
            for(Integer iIdx : entry.getValue()) {
                _Doc temp = m_bipartite.getCorpus().getCollection().get(m_reviewIndex.get(String.format("%d_%d", iIdx, entry.getKey())));
                docs.add(temp);
            }
        }
        String rateFile = String.format("%s/%s/%d/%s.tsv", prefix, source, k, mode);
        String userFile = String.format("%s/%s/%d/%s_users.tsv", prefix, source, k, mode);
        String docFile = String.format("%s/%s/%d/mult.dat", prefix, source, k);

        (new File(rateFile)).getParentFile().mkdirs();
        saveRate(rateFile, docs);
        saveUser(userFile, docs);

        if(mode.equals("train")) {
            docs.clear();
            docs = buildItemProfile(m_bipartite.getCorpus().getCollection());
            saveDoc(docFile, docs);
        }
    }

    //build query for one item from its' reviews
    public static ArrayList<_Doc> buildItemProfile(ArrayList<_Doc> docs){
        ArrayList<_Doc> itemProfile = new ArrayList<>();

        //allocate review by item
        for(_Doc doc : docs){
            String itemID = doc.getItemID();
            ((_Item) m_items.get(m_itemsIndex.get(itemID))).addOneReview((_Review)doc);
        }
        //compress sparse feature of reviews into one vector for each item
        int i = 0;
        for(_Product item : m_items){
            ((_Item) item).buildProfile("");
            _Doc itemDoc = new _Doc(i++, "", 0);
            itemDoc.createSpVct(((_Item) item).getFeature());
            itemProfile.add(itemDoc);
        }

        return itemProfile;
    }


    public static void save2FileRTM(String prefix, String source, String mode, int k) throws IOException{
        ArrayList<_Doc> docs = new ArrayList<>();
        ArrayList<int[]> links = new ArrayList<>();
        for(Map.Entry<Integer, ArrayList<Integer>> entry : m_mapByUser.entrySet()){
            ArrayList<Integer> cluster = new ArrayList<>();
            for(Integer iIdx : entry.getValue()) {
                _Doc temp = m_bipartite.getCorpus().getCollection().get(m_reviewIndex.get(String.format("%d_%d", iIdx, entry.getKey())));
                cluster.add(docs.size());
                docs.add(temp);
            }
            for(int i = 0; i < cluster.size(); i++){
                for(int j = i+1; j < cluster.size(); j++){
                    links.add(new int[]{i,j});
                }
            }
        }
        String docFile = String.format("%s/%s_user_corpus_%s_%d.txt", prefix,source, mode, k);
        String linkFile = mode.equals("test") ? String.format("%s/%s_user_link_%s_train_%d.txt", prefix, source, mode, k) : String.format("%s/%s_user_link_%s_%d.txt", prefix, source, mode, k);
        saveDoc(docFile, docs);
        saveLink(linkFile, links);
        if(mode.equals("test"))
            (new File(String.format("%s/%s_user_link_%s_test_%d.txt", prefix, source, mode, k))).createNewFile();

        for(Map.Entry<Integer, ArrayList<Integer>> entry : m_mapByItem.entrySet()){
            ArrayList<Integer> cluster = new ArrayList<>();
            for(Integer uIdx : entry.getValue()) {
                _Doc temp = m_bipartite.getCorpus().getCollection().get(m_reviewIndex.get(String.format("%d_%d", entry.getKey(), uIdx)));
                cluster.add(docs.size());
                docs.add(temp);
            }
            for(int i = 0; i < cluster.size(); i++){
                for(int j = i+1; j < cluster.size(); j++){
                    links.add(new int[]{i,j});
                }
            }
        }
        docFile = String.format("%s/%s_item_corpus_%s_%d.txt", prefix,source, mode, k);
        linkFile = mode.equals("test") ? String.format("%s/%s_item_link_%s_train_%d.txt", prefix, source, mode, k) : String.format("%s/%s_item_link_%s_%d.txt", prefix, source, mode, k);
        saveDoc(docFile, docs);
        saveLink(linkFile, links);
        if(mode.equals("test"))
            (new File(String.format("%s/%s_item_link_%s_test_%d.txt", prefix, source, mode, k))).createNewFile();
    }

    public static void saveRate(String filename, ArrayList<_Doc> docs) {
        if (filename==null || filename.isEmpty()) {
            System.out.println("Please specify the file name to save the vectors!");
            return;
        }

        try {
            BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(filename)));
            for(_Doc doc:docs) {
                _Review r = (_Review) doc;
                String userID = r.getUserID();
                String itemID = r.getItemID();
                int rate = r.getYLabel();
                writer.write(String.format("%d\t%d\t%d\n", m_usersIndex.get(userID), m_itemsIndex.get(itemID), rate));
            }
            writer.close();

            System.out.format("[Info]%d rates saved to %s\n", docs.size(), filename);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static void saveUser(String filename, ArrayList<_Doc> docs) {
        if (filename==null || filename.isEmpty()) {
            System.out.println("Please specify the file name to save the vectors!");
            return;
        }

        try {
            BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(filename)));
            Set<String> users = new HashSet<>();
            for(_Doc doc:docs) {
                _Review r = (_Review) doc;
                String userID = r.getUserID();
                users.add(userID);
            }
            for(String uid : users)
                writer.write(String.format("%d\n", m_usersIndex.get(uid)));
            writer.close();

            System.out.format("[Info]%d users saved to %s\n", users.size(), filename);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static void saveDoc(String filename, ArrayList<_Doc> docs) {
        if (filename==null || filename.isEmpty()) {
            System.out.println("Please specify the file name to save the vectors!");
            return;
        }

        try {
            BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(filename)));
            for(_Doc doc:docs) {
                writer.write(String.format("%d", doc.getTotalDocLength()));
                for(_SparseFeature fv:doc.getSparse())
                    writer.write(String.format(" %d:%.1f", fv.getIndex(), fv.getValue()));//index starts from 1
                writer.write("\n");
            }
            writer.close();

            System.out.format("[Info]%d feature vectors saved to %s\n", docs.size(), filename);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static void saveLink(String filename, ArrayList<int[]> links){
        if (filename==null || filename.isEmpty()) {
            System.out.println("Please specify the file name to save the vectors!");
            return;
        }

        try {
            BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(filename)));
            for(int[] lk:links) {
                writer.write(String.format("%d\t%d\n", lk[0], lk[1]));//index starts from 1
            }
            writer.close();

            System.out.format("[Info]%d links saved to %s\n", links.size(), filename);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
