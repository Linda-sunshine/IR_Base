package Analyzer;

import structures.*;

import java.io.File;
import java.io.FileWriter;
import java.util.*;

public class BipartiteAnalyzer {
    int m_k;
    _Corpus m_corpus;
    ArrayList<_Doc> m_trainSet;
    ArrayList<_Doc> m_testSet;

    protected List<_User> m_users;
    protected List<_Product> m_items;

    protected HashMap<String, Integer> m_usersIndex; //(userID, index in m_users)
    protected HashMap<String, Integer> m_itemsIndex; //(itemID, index in m_items)
    protected HashMap<String, Integer> m_reviewIndex; //(itemIndex_userIndex, index in m_corpus.m_collection)

    protected HashMap<Integer, ArrayList<Integer>>  m_mapByUser; //train
    protected HashMap<Integer, ArrayList<Integer>> m_mapByItem;
    protected HashMap<Integer, ArrayList<Integer>>  m_mapByUser_global; //global
    protected HashMap<Integer, ArrayList<Integer>> m_mapByItem_global;
    protected HashMap<Integer, ArrayList<Integer>> m_mapByUser_test; //test
    protected HashMap<Integer, ArrayList<Integer>> m_mapByItem_test;

    public BipartiteAnalyzer(_Corpus corpus){
        this.m_corpus = corpus;
        m_users = new ArrayList<>();
        m_items = new ArrayList<>();
        m_usersIndex = new HashMap<>();
        m_itemsIndex = new HashMap<>();
        m_reviewIndex = new HashMap<>();
        m_mapByUser = new HashMap<>();
        m_mapByItem = new HashMap<>();
        m_mapByUser_test = new HashMap<>();
        m_mapByItem_test = new HashMap<>();
        m_mapByUser_global = new HashMap<>();
        m_mapByItem_global = new HashMap<>();
    }

    public void reset(){
        this.m_corpus.reset();
        m_users.clear();
        m_items.clear();
        m_usersIndex.clear();
        m_itemsIndex.clear();
        m_reviewIndex.clear();
        m_mapByUser.clear();
        m_mapByItem.clear();
        m_mapByUser_test.clear();
        m_mapByItem_test.clear();
        m_mapByUser_global.clear();
        m_mapByItem_global.clear();
    }

    public void analyzeCorpus(){
        System.out.println("[Info]Analzying corpus: ");

        m_users.clear();
        m_items.clear();
        m_usersIndex.clear();
        m_itemsIndex.clear();
        m_reviewIndex.clear();
        int u_index = 0, i_index = 0, size = m_corpus.getCollection().size();
        for(int d = 0; d < size; d++){
            _Review doc = (_Review) m_corpus.getCollection().get(d);
            String userID = doc.getUserID();
            String itemID = doc.getItemID();

            if(!m_usersIndex.containsKey(userID)){
                m_users.add(new _User4ETBIR(userID));
                m_usersIndex.put(userID, u_index);
                u_index++;
            }

            if(!m_itemsIndex.containsKey(itemID)){
                m_items.add(new _Product4ETBIR(itemID));
                m_itemsIndex.put(itemID, i_index);
                i_index++;
            }

            int uIdx = m_usersIndex.get(userID);
            int iIdx = m_itemsIndex.get(itemID);
            m_reviewIndex.put(iIdx + "_" + uIdx, d);

            //if ( (100 * d/size) % 10 == 0 )
            if(d % (size/10) == 0)
                System.out.print(".");//every 10%
        }

        System.out.format("-- Global corpus: vocabulary size: %d, review size: %d, item size: %d, user size: %d\n",
                m_corpus.getFeatureSize(), size,  m_items.size(),  m_users.size());
    }

    public boolean analyzeBipartite(ArrayList<_Doc> docs, String source){
        HashMap<Integer, ArrayList<Integer>> mapByUser = new HashMap<>();
        HashMap<Integer, ArrayList<Integer>> mapByItem = new HashMap<>();
        if(source.equals("train")){
            mapByUser = m_mapByUser;
            mapByItem = m_mapByItem;
        }else if(source.equals("test")){
            mapByUser = m_mapByUser_test;
            mapByItem = m_mapByItem_test;
        }else{
            mapByUser = m_mapByUser_global;
            mapByItem = m_mapByItem_global;
        }

        System.out.format("[Info]Analying bipartie graph: \n");
        mapByItem.clear();
        mapByUser.clear();

        if(m_usersIndex == null){
            System.err.println("[Warning]Analyze Corpus first! Analyzing...");
            analyzeCorpus();
        }

        for (_Doc doc:docs){
            _Review d = (_Review) doc;
            int u_index = m_usersIndex.get(d.getUserID());
            int i_index = m_itemsIndex.get(d.getItemID());
            if(!mapByUser.containsKey(u_index)){
                mapByUser.put(u_index, new ArrayList<Integer>());
            }
            if(!mapByItem.containsKey(i_index)){
                mapByItem.put(i_index, new ArrayList<Integer>());
            }
            mapByUser.get(u_index).add(i_index);
            mapByItem.get(i_index).add(u_index);
        }
        System.out.format("-- %s graph: review size: %d, item size: %d, user size: %d\n",
                source, docs.size(), mapByItem.size(), mapByUser.size());

        return true;
    }

    public boolean splitCorpusColdStart(int k, String outFolder){
        System.out.format("[Info]Splitting corpus into %d folds: ", m_k);

        this.m_k = k;
        m_trainSet = new ArrayList<>();
        m_testSet = new ArrayList<>();

        ArrayList<_Doc> docs = m_corpus.getCollection();

        if(m_usersIndex == null){
            System.err.println("[Warning]Analysing corpus first! Analyzing with cold start...");
            analyzeCorpus();
        }

        analyzeBipartite(docs, "global");

        //for cold start user
        Random rand = new Random();
        int[] mask_user = new int[m_users.size()];
        for(int i=0; i< mask_user.length; i++) {
            mask_user[i] = rand.nextInt(k);
        }
        //inverted index
        HashMap<Integer, ArrayList<Integer>> divid_user = new HashMap<>();//key: mask, value: idx of user
        for(int i = 0; i < mask_user.length; i++){
            if(!divid_user.containsKey(mask_user[i]))
                divid_user.put(mask_user[i], new ArrayList<Integer>());
            divid_user.get(mask_user[i]).add(i);
        }
        //half index for each mask should be disabled for other mask
        HashMap<Integer, HashSet<Integer>> exclusive_user = new HashMap<>();
        for(Integer mask : divid_user.keySet()){
            int half_mark = (divid_user.get(mask).size()+1) / 2;
            exclusive_user.put(mask, new HashSet<Integer>());
            for(int i = 0; i < half_mark; i++){
                exclusive_user.get(mask).add(divid_user.get(mask).get(i));
            }
        }

        //for cold start item
        Random rand2 = new Random();
        int[] mask_item = new int[m_items.size()];
        for(int i=0; i< mask_item.length; i++) {
            mask_item[i] = rand2.nextInt(k);
        }
        //inverted index
        HashMap<Integer, ArrayList<Integer>> divid_item = new HashMap<>();//key: mask, value: idx of user
        for(int i = 0; i < mask_item.length; i++){
            if(!divid_item.containsKey(mask_item[i]))
                divid_item.put(mask_item[i], new ArrayList<Integer>());
            divid_item.get(mask_item[i]).add(i);
        }
        //half index for each mask should be disabled for other mask
        HashMap<Integer, HashSet<Integer>> exclusive_item = new HashMap<>();
        for(Integer mask : divid_item.keySet()){
            int half_mark = (divid_item.get(mask).size()+1) / 2;
            exclusive_item.put(mask, new HashSet<Integer>());
            for(int i = 0; i < half_mark; i++){
                exclusive_item.get(mask).add(divid_item.get(mask).get(i));
            }
        }

        //Use this loop to iterate all the ten folders, set the train set and test set.
        int[] label = new int[docs.size()];
        Arrays.fill(label, 0);
        for (int i = 0; i < m_k; i++) {
            //cold start item
            ArrayList<Integer> user_valid = new ArrayList<>();
            for (int j = 0; j < mask_item.length; j++) {
                if( mask_item[j]==i
                        || (i!=m_k-1 && mask_item[j]==i+1 && !exclusive_item.get(i+1).contains(j))
                        || (i==m_k-1 && mask_item[j]==0 && !exclusive_item.get(0).contains(j))) {
                    user_valid.add(j);
                }
            }
            //cold start user
            ArrayList<Integer> item_valid = new ArrayList<>();
            for (int j = 0; j < mask_user.length; j++) {
                if( mask_user[j]==i
                        || (i!=m_k-1 && mask_user[j]==i+1 && !exclusive_user.get(i+1).contains(j))
                        || (i==m_k-1 && mask_user[j]==0 && !exclusive_user.get(0).contains(j))) {
                    item_valid.add(j);
                }
            }
            //test
            for(Integer iIdx : item_valid){
                for(Integer uIdx : user_valid){
                    if(m_reviewIndex.containsKey(String.format("%d_%d", iIdx, uIdx))){
                        int docIdx = m_reviewIndex.get(String.format("%d_%d", iIdx, uIdx));
                        m_testSet.add(docs.get(docIdx));
                        label[docIdx] = 1;
                    }
                }
            }
            //rest is filled with random doc
            for (int j = 0; j < docs.size(); j++) {
                if(label[j]>=1)
                    continue;
                else
                    m_trainSet.add(docs.get(j));
            }

            // generate bipartie for training set
            analyzeBipartite(m_trainSet, "train");

            // generate bipartie for testing set
            analyzeBipartite(m_testSet, "test");
            save2File(outFolder, String.valueOf(i));

            System.out.format("-- Fold No. %d: train size = %d, test size = %d, cold user size = %d, cold item size = %d\n",
                    i, m_trainSet.size(), m_testSet.size(), exclusive_user.get(i).size(), exclusive_item.get(i).size());
            m_trainSet.clear();
            m_testSet.clear();
        }

        return true;
    }

    public boolean splitCorpus(int k, String outFolder) {
        System.out.format("[Info]Splitting corpus into %d folds: ", m_k);

        this.m_k = k;
        m_trainSet = new ArrayList<>();
        m_testSet = new ArrayList<>();

        m_corpus.shuffle(m_k);
        int[] masks = m_corpus.getMasks();
        ArrayList<_Doc> docs = m_corpus.getCollection();

        if(m_usersIndex == null){
            System.err.println("[Warning]Analysing corpus first! Analyzing...");
            analyzeCorpus();
        }

        //Use this loop to iterate all the ten folders, set the train set and test set.
        for (int i = 0; i < m_k; i++) {
            for (int j = 0; j < masks.length; j++) {
                if( masks[j]==i )
                    m_testSet.add(docs.get(j));
                else
                    m_trainSet.add(docs.get(j));
            }

            // generate bipartie for training set
            analyzeBipartite(m_trainSet, "train");

            // generate bipartie for testing set
            analyzeBipartite(m_testSet, "test");
            save2File(outFolder, String.valueOf(i));

            System.out.format("-- Fold No. %d: train size = %d, test size = %d\n", i, m_trainSet.size(), m_testSet.size());
            m_trainSet.clear();
            m_testSet.clear();
        }

        return true;
    }

    public boolean deleteDir(String outFolder){
        File dir = new File(outFolder);
        if(dir.isDirectory()){
            String[] children = dir.list();
            for(int i=0; i<children.length; i++){
                boolean success = deleteDir(children[i]);
                if(!success){
                    return false;
                }
            }
        }
        return dir.delete();
    }

    public void save2File(String outFolder, String mode){
        try {
            // save train
            String outTrain = outFolder + mode + "/";
            new File(outTrain).mkdirs();
            HashMap<Integer, ArrayList<Integer>> mapByUser = mode.equals("train")?m_mapByUser:m_mapByUser_test;
            for (int u_idx : mapByUser.keySet()) {
                _User user = m_users.get(u_idx);
                String userID = user.getUserID();
                FileWriter file = new FileWriter(outTrain + userID + ".txt");
                file.write(userID + "\n");
                for (int i_idx : mapByUser.get(u_idx)){
                    _Product item = m_items.get(i_idx);
                    _Doc doc =  m_corpus.getCollection().get(m_reviewIndex.get(i_idx + "_" + u_idx));
                    file.write(item.getID() + "\n");
                    file.write(doc.getSource() + "\n");
                    file.write("\n");
                    file.write(doc.getYLabel() + "\n");
                    file.write(doc.getTimeStamp() + "\n");
                }
                file.flush();
                file.close();
            }
        }catch (Exception e){
            e.printStackTrace();
        }
    }

    public _Corpus getCorpus(){ return this.m_corpus; }
    public List<_User> getUsers(){ return this.m_users; }
    public List<_Product> getItems(){ return this.m_items; }
    public HashMap<String, Integer> getUsersIndex() { return this.m_usersIndex; }
    public HashMap<String, Integer> getItemsIndex() {return this.m_itemsIndex; }
    public HashMap<String, Integer> getReviewIndex() {return this.m_reviewIndex; }
    public HashMap<Integer, ArrayList<Integer>> getMapByUser() { return this.m_mapByUser; }
    public HashMap<Integer, ArrayList<Integer>> getMapByItem() { return this.m_mapByItem; }
    public HashMap<Integer, ArrayList<Integer>> getMapByUser_test(){ return this.m_mapByUser_test; }
    public HashMap<Integer, ArrayList<Integer>> getMapByItem_test() { return this.m_mapByItem_test; }
}
