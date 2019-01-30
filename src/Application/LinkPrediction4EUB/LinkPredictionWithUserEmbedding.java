package Application.LinkPrediction4EUB;

import Application.LinkPrediction4MMB.LinkPrediction;
import utils.Utils;

import java.io.*;
import java.util.*;

/**
 * @author Lin Gong (lg5bt@virginia.edu)
 */
public class LinkPredictionWithUserEmbedding {

    protected class _Object4Link {
        String m_id;
        int m_index;
        ArrayList<_Edge4Link> m_edges;

        // this is for question ranking
        public _Object4Link(String id){
            m_id = id;
            m_edges = new ArrayList<>();
        }
        // this is for user ranking
        public _Object4Link(String id, int idx){
            m_id = id;
            m_index = idx;
            m_edges = new ArrayList<_Edge4Link>();
        }

        public int getIndex(){
            return m_index;
        }
        public void addOneEdge(_Edge4Link e){
            m_edges.add(e);
        }

        public ArrayList<_Edge4Link> getEdges(){
            return m_edges;
        }

    }

    class _Edge4Link {
        int m_index;
        double m_sim;
        int m_label;

        public _Edge4Link(int idx, double sim, int label) {
            m_index = idx;
            m_sim = sim;
            m_label = label;
        }

        public int getLabel() {
            return m_label;
        }
    }

    int m_userSize, m_dim;
    double[] m_NDCGs, m_MAPs;

    protected double[][] m_similarity;
    protected double[][] m_embeddings;
    protected ArrayList<String> m_userIds;
    protected ArrayList<String> m_testIds;
    protected HashMap<String, Integer> m_idIndexMap;
    protected HashMap<String, _Object4Link> m_testMap;
    protected Object m_NDCGMAPLock = null;

    public LinkPredictionWithUserEmbedding(){
        m_idIndexMap = new HashMap<>();
        m_testMap = new HashMap<>();
        m_NDCGMAPLock = new Object();
        m_testIds = new ArrayList<>();
    }

    // load user ids for later use
    public void loadUserIds(String idFile){
        try {
            m_userIds = new ArrayList<>();
            File file = new File(idFile);
            BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(file), "UTF-8"));
            String line;

            while ((line = reader.readLine()) != null) {
                String uid = line.trim();
                m_idIndexMap.put(uid, m_userIds.size());
                m_userIds.add(uid);
            }
            m_userSize = m_userIds.size();
            System.out.format("Finish loading %d user ids from %s.\n", m_userIds.size(), idFile);
        }catch(IOException e){
            e.printStackTrace();
        }
    }
    // load each user's embedding
    public void loadUserEmbedding(String filename){
        try {
            File file = new File(filename);
            BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(file), "UTF-8"));
            String line;

            String firstLine = reader.readLine();
            String[] strs = firstLine.trim().split("\\s+");
            if(strs.length != 2){
                System.out.println("[error]The file is not correct!! Double check user embedding file!");
                return;
            }
            int userSize = Integer.valueOf(strs[0]);
            m_dim = Integer.valueOf(strs[1]);
            m_embeddings = new double[m_userIds.size()][m_dim];
            if(userSize > m_userIds.size()) {
                System.out.println("[error]The file is not correct!! Double check user embedding file!");
                return;
            }
            // read each user's embedding one by one
            int count = 0;
            while ((line = reader.readLine()) != null) {
                if(count > m_userSize){
                    System.out.println("[error]The line number exceeds the user size!!");
                    break;
                }
                String[] valStrs = line.trim().split("\\s+");
                if(valStrs.length != m_dim + 1){
                    System.out.println("[error]The user's dimension is not correct!!");
                    continue;
                }
                String uid = valStrs[0];
                double[] embedding = new double[m_dim];
                for(int i=1; i<valStrs.length; i++){
                    embedding[i-1] = Double.valueOf(valStrs[i]);
                }
                int index = m_idIndexMap.get(uid);
                m_embeddings[index] = embedding;
                count++;
            }
            reader.close();
            System.out.format("[Info]Finish loading %d user embeddings from %s.\n", count, filename);
        } catch(IOException e){
            e.printStackTrace();
        }
    }

    // load edges (interactions + non-interactions) for all the users
    public void loadTestOneZeroEdges(String filename){
        int labelOne = 1, labelZero = 0, count = 0;
        try {
            m_testMap.clear();
            m_testIds.clear();
            File file = new File(filename);
            BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(file), "UTF-8"));
            String line;
            HashSet<Integer> testOneIndexes = new HashSet<>();
            while ((line = reader.readLine()) != null) {
                testOneIndexes.clear();
                String[] strs = line.trim().split("\t");
                String id = strs[0];
                int uiIdx = m_idIndexMap.get(id);
                _Object4Link ui = new _Object4Link(strs[0], uiIdx);
                m_testMap.put(id, ui);
                m_testIds.add(id);
                // record testing interactions of ui or qi
                for(int j=1; j<strs.length; j++){
                    String ujId = strs[j];
                    int ujIdx = m_idIndexMap.get(ujId);
                    testOneIndexes.add(ujIdx);
                    if(ui.getIndex() != uiIdx)
                        System.err.println("Index not aligned!!!");
                    double sim = getSimilarity(ui.getIndex(), ujIdx);
                    ui.addOneEdge(new _Edge4Link(ujIdx, sim, labelOne));
                }
                // add the test non-interactions of ui
                for(int j=0; j<m_idIndexMap.size(); j++){
                    if(j == uiIdx || testOneIndexes.contains(j)) continue;
                    double sim = getSimilarity(uiIdx, j);
                    ui.addOneEdge(new _Edge4Link(j, sim, labelZero));
                }
                count++;
            }
            reader.close();
            System.out.format("[Info]%d/%d users' interactions are loaded!!\n", m_testMap.size(), count);
        } catch(IOException e){
            e.printStackTrace();
        }
    }

    // load interactions for all the users
    public void loadTestOneEdges(String filename){
        int labelOne = 1, count = 0;
        try {
            m_testMap.clear();
            m_testIds.clear();
            File file = new File(filename);
            BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(file), "UTF-8"));
            String line;
            while ((line = reader.readLine()) != null) {
                String[] strs = line.trim().split("\t");
                String uid = strs[0];
                int index = m_idIndexMap.get(uid);
                _Object4Link ui = new _Object4Link(strs[0], index);
                m_testMap.put(uid, ui);
                m_testIds.add(uid);
                for(int j=1; j<strs.length; j++){
                    String ujId = strs[j];
                    int ujIdx = m_idIndexMap.get(ujId);
                    double sim = getSimilarity(ui.getIndex(), ujIdx);
                    ui.addOneEdge(new _Edge4Link(ujIdx, sim, labelOne));
                }
                count++;
            }
            reader.close();
            System.out.format("[Info]%d/%d users' interactions are loaded!!\n", m_testMap.size(), count);
        } catch(IOException e){
            e.printStackTrace();
        }
    }

    // load non-interactions from file for all the users
    public void loadTestZeroEdges(String filename){
        int labelZero = 0, count = 0;
        try {
            File file = new File(filename);
            BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(file), "UTF-8"));
            String line;
            while ((line = reader.readLine()) != null) {
                String[] strs = line.trim().split("\t");
                String uid = strs[0];
                if(!m_testMap.containsKey(uid))
                    continue;
                _Object4Link ui = m_testMap.get(uid);
                for(int j=1; j<strs.length; j++){
                    String ujId = strs[j];
                    int ujIdx = m_idIndexMap.get(ujId);
                    double sim = getSimilarity(ui.getIndex(), ujIdx);
                    ui.addOneEdge(new _Edge4Link(ujIdx, sim, labelZero));
                }
                count++;
            }
            reader.close();
            System.out.format("[Info]%d/%d users' non-interactions are loaded!!\n", m_testMap.size(), count);
        } catch(IOException e){
            e.printStackTrace();
        }
    }

    public double getSimilarity(int i, int j){
        if(i == j){
            System.out.println("[bug]Same i and j for similarity!!");
            return 0;
        } else if(i < j)
            return m_similarity[i][j];
        else
            return m_similarity[j][i];
    }

    public void calcSimilarity(){

        System.out.println("Start calculating similarity....");
        m_similarity = new double[m_userSize][m_userSize];
        for(int i=0; i<m_userSize; i++){
            for(int j=i+1; j<m_userSize; j++){
//                m_similarity[i][j] = Math.random();
                m_similarity[i][j] = Utils.cosine(m_embeddings[i], m_embeddings[j]);
            }
        }
        System.out.println("Finish calculating similarity.");

    }
    public void saveUserIds(String embedFile, String idFile){
        try {
            File file = new File(embedFile);
            BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(file), "UTF-8"));
            String line;

            ArrayList<String> uids = new ArrayList<>();
            String firstLine = reader.readLine();
            String[] strs = firstLine.trim().split("\t");
            if(strs.length != 2){
                System.out.println("[error]The file is not correct!! Double check user embedding file!");
                return;
            }
            int userSize = Integer.valueOf(strs[0]);
            // read each user's embedding one by one
            while ((line = reader.readLine()) != null) {

                String[] valStrs = line.trim().split("\t");
                String uid = valStrs[0];
                uids.add(uid);
            }
            reader.close();
            System.out.format("[Info]Finish loading %d user ids from %s.\n", uids.size(), embedFile);
            PrintWriter writer = new PrintWriter(new File(idFile));
            for(String id: uids)
                writer.write(id + "\n");
            writer.close();
            System.out.format("[Info]Finish saving %d user ids to %s.\n", uids.size(), idFile);

        } catch(IOException e){
            e.printStackTrace();
        }
    }

    public void initLinkPred(String idFile, String embedFile, String testInterFile, String testNonInterFile) {
        loadUserIds(idFile);
        loadUserEmbedding(embedFile);
        calcSimilarity();
        // if no non-interactions are specified, we will consider all the other users as non-interactions
        // thus, non-interactions are know after loading interactions
        if(testNonInterFile == null){
            loadTestOneZeroEdges(testInterFile);
        } else{
            // otherwise, load one-edges ans zero-edges individually
            loadTestOneEdges(testInterFile);
            loadTestZeroEdges(testNonInterFile);
        }

    }


    // The function for calculating all NDCGs and MAPs.
    public void calculateAllNDCGMAP(){
        m_NDCGs = new double[m_testIds.size()];
        m_MAPs = new double[m_testIds.size()];
        Arrays.fill(m_NDCGs, -1);
        Arrays.fill(m_MAPs, -1);

        System.out.print("[Info]Start calculating NDCG and MAP...\n");
        int numberOfCores = Runtime.getRuntime().availableProcessors();
        ArrayList<Thread> threads = new ArrayList<Thread>();

        for(int k=0; k<numberOfCores; ++k){
            threads.add((new Thread() {
                int core, numOfCores;
                @Override
                public void run() {
                    try {

                        for (int i = 0; i + core <m_testIds.size(); i += numOfCores) {
                            if(i%500==0) System.out.print(".");
                            String uid = m_testIds.get(i + core);
                            _Object4Link user = m_testMap.get(uid);
                            double[] vals = calculateNDCGMAP(user);
                            // put the calculated nDCG into the array for average calculation
                            synchronized(m_NDCGMAPLock){
                                m_NDCGs[i+core] = vals[0];
                                m_MAPs[i+core] = vals[1];
                            }
                        }
                    } catch(Exception ex) {
                        ex.printStackTrace();
                    }
                }

                private Thread initialize(int core, int numOfCores) {
                    this.core = core;
                    this.numOfCores = numOfCores;
                    return this;
                }
            }).initialize(k, numberOfCores));
            threads.get(k).start();
        }

        for(int k=0;k<numberOfCores;++k){
            try {
                threads.get(k).join();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
    }

//    // The function for calculating all NDCGs and MAPs.
//    public void calculateAllNDCGMAP(){
//        m_NDCGs = new double[m_testUserIds.size()];
//        m_MAPs = new double[m_testUserIds.size()];
//        Arrays.fill(m_NDCGs, -1);
//        Arrays.fill(m_MAPs, -1);
//
//        System.out.print("[Info]Start calculating NDCG and MAP...\n");
//
//        for (int i = 0; i <m_testUserIds.size(); i++) {
//            String uid = m_testUserIds.get(i);
//            _User4Link user = m_testUserMap.get(uid);
//            double[] vals = calculateNDCGMAP(user);
//            // put the calculated nDCG into the array for average calculation
//            m_NDCGs[i] = vals[0];
//            m_MAPs[i] = vals[1];
//        }
//    }

    // calculate NDCG and MAP for one user
    public double[] calculateNDCGMAP(_Object4Link user){

        // As we load the one edges first, then zero edges
        double iDCG = 0, DCG = 0, PatK = 0, AP = 0, count = 0;
        ArrayList<_Edge4Link> candidates = user.getEdges();

        //Calculate iDCG
        for(int i=0; i<candidates.size(); i++) {
            //log(i+1), since i starts from 0, add 1 more.
            iDCG += (Math.pow(2, candidates.get(i).getLabel()) - 1) / (Math.log(i + 2));
        }

        // Sort the candidates and calculate the DCG, nDCG = DCG/iDCG.
        Collections.sort(candidates, new Comparator<_Edge4Link>(){
            @Override
            public int compare(_Edge4Link e1, _Edge4Link e2){
                if(e1.m_sim  < e2.m_sim)
                    return 1;
                else if(e1.m_sim > e2.m_sim)
                    return -1;
                else{
                    return 0;
                }
            }
        });

        for(int i=0; i<candidates.size(); i++){
            _Edge4Link edge = candidates.get(i);
            DCG += (Math.pow(2, edge.getLabel()) -1)/(Math.log(i+2));
            if(edge.getLabel() == 1){
                PatK = (count+1)/((double)i+1);
                AP += PatK;
                count++;
            }
        }
		if(Double.isNaN(DCG/iDCG))
			System.out.println("[error] Nan NDCG! Debug here!!");
        return new double[]{DCG/iDCG, AP/count};
    }

    public double[] calculateAvgNDCGMAP(){
        double avgNDCG = 0, avgMAP = 0;
        int valid = 0;
        for(int i=0; i<m_NDCGs.length; i++){
            if(m_NDCGs[i] == -1 || m_MAPs[i] == -1 || Double.isNaN(m_NDCGs[i]) || Double.isNaN(m_MAPs[i]))
                continue;
            valid++;
            avgNDCG += m_NDCGs[i];
            avgMAP += m_MAPs[i];
        }
        avgNDCG /= valid;
        avgMAP /= valid;
        System.out.format("\n[Info]Valid user size: %d, Avg NDCG, MAP -- %.5f\t%.5f\n\n", valid, avgNDCG, avgMAP);
        return new double[]{avgNDCG, avgMAP};
    }

//    public static void main(String[] args){
//        String data = "YelpNew";
//        for(int dim: new int[]{10}) {
//
//            for (int fold : new int[]{0}) {
//                int[] times = new int[]{2, 3, 4, 5};
//                String[] models = new String[]{"EUB", "LDA", "HFT", "TADW"}; // "LDA", "HFT", "TADW", "EUB", "LDA", "HFT",
//
//                String prefix = "";
//                double[][][] perfs = new double[models.length][times.length][2];
//                for (int t = 0; t < times.length; t++) {
//                    int time = times[t];
//                    for (int i = 0; i < models.length; i++) {
//                        String model = models[i];
//                        System.out.format("-----current model-%s-time-%d-dim-%d------\n", model, time, dim);
//
//                        String idFile = String.format("/Users/lin/DataWWW2019/UserEmbedding/%s_userids.txt", data);
//                        String embedFile = String.format("/Users/lin/DataWWW2019/UserEmbedding/%s_%s_embedding_dim_%d_fold_%d.txt", data, model, dim, fold);
//                        String testInterFile = String.format("./data/DataEUB/CV4Edges/%sCVIndex4Interaction_fold_%d_test.txt", data, fold);
//                        String testNonInterFile = String.format("./data/DataEUB/CV4Edges/%sCVIndex4NonInteraction_time_%d_fold_%d.txt", data, time, fold);
//
//                        LinkPredictionWithUserEmbedding link = new LinkPredictionWithUserEmbedding();
////                        link.saveUserIds(embedFile, idFile);
//                        link.ininLinkPred(idFile, embedFile, testInterFile, testNonInterFile);
//                        link.calculateAllNDCGMAP();
//                        perfs[i][t] = link.calculateAvgNDCGMAP();
//
//                    }
//                }
//                for (int time : times) {
//                    System.out.format("\t\t%d\t\t", time);
//                }
//                System.out.println();
//                for (int i = 0; i < models.length; i++) {
//                    System.out.print(models[i] + "\t");
//                    for (double[] ndcgMap : perfs[i]) {
//                        System.out.format("%.4f\t%.4f\t", ndcgMap[0], ndcgMap[1]);
//                    }
//                    System.out.println();
//                }
//            }
//        }
//    }

    public void addTwoArrays(double[][] a, double[][] b){
        for(int i=0; i<a.length; i++){
            for(int j=0; j<a[0].length; j++){
                a[i][j] += b[i][j];
            }
        }
    }

    public void calcMeanStd(HashMap<String, double[][][]> map){
        HashMap<String, double[][]> meanMap = new HashMap<>();
        HashMap<String, double[][]> stdMap = new HashMap<>();

        for(String model: map.keySet()){
            double[][][] perfs = map.get(model);

            double folds = perfs.length;
            int times = perfs[0].length;
            double[][] mean = new double[times][2];
            double[][] std = new double[times][2];

            for (int i = 0; i < folds; i++) {
                addTwoArrays(mean, perfs[i]);
            }
            // calc mean
            for (int i = 0; i < times; i++) {
                for (int j = 0; j < 2; j++) {
                    mean[i][j] /= folds;
                }
            }

            // calc std
            for (int i = 0; i < folds; i++) {
                for (int j = 0; j < times; j++) {
                    for (int m = 0; m < 2; m++) {
                        std[j][m] += (perfs[i][j][m] - mean[j][m]) * (perfs[i][j][m] - mean[j][m]);
                    }
                }
            }
            for (int j = 0; j < times; j++) {
                for (int m = 0; m < 2; m++) {
                    std[j][m] = Math.sqrt(std[j][m] / folds);
                }
            }
            meanMap.put(model, mean);
            stdMap.put(model, std);
        }
        printMeanStd(meanMap, stdMap);
    }

    public void printMeanStd(HashMap<String, double[][]> meanMap, HashMap<String, double[][]> stdMap){
        System.out.print("\tNDCG\tMAP\tNDCG\tMAP\tNDCG\tMAP\tNDCG\tMAP\n");
        for(String model: meanMap.keySet()){
            System.out.print(model+"\t");
            double[][] mean = meanMap.get(model);
            double[][] std = stdMap.get(model);
            for(int t=0; t<mean.length; t++){
                System.out.format("%.5f+/-%.5f\t%.5f+/-%.5f\t", mean[t][0], std[t][0], mean[t][1], std[t][1]);
            }
            System.out.println();
        }
    }
    public static void main(String[] args){

        String data = "YelpNew";
        String prefix = "/home/lin"; // "/Users/lin", "/home/lin"

        int dim = 10, folds = 4;
        String idFile = String.format("%s/DataWWW2019/UserEmbedding/%s_userids.txt", prefix, data);

        int[] times = new int[]{2, 3, 4, 5, 6, 7, 8};
        String[] models = new String[]{"DW"};//, "EUB_t40-0.5", "EUB_t40-1", "EUB_t40-2"}; //"BOW", "LDA", "HFT", "RTM", "DW", "TADW", "EUB_t10", "EUB_t20", "EUB_t30", "EUB_t40", "EUB_t50"};// "RTM", "LDA", "HFT", "DW", "TADW"}; // "LDA", "HFT", "TADW", "EUB", "LDA", "HFT"
        HashMap<String, double[][][]> allFoldsPerf = new HashMap<String, double[][][]>();

        LinkPredictionWithUserEmbedding link = null;

        for(String model: models){
            if(model.equals("LDA") || model.equals("HFT") || model.equals("BOW"))
                folds = 0;
            double[][][] perfs = new double[folds+1][times.length][2];
            for (int t = 0; t < times.length; t++) {

                for(int fold=0; fold<=folds; fold++){
                    int time = times[t];
                    System.out.format("-----current model-%s-time-%d-dim-%d------\n", model, time, dim);

                    String embedFile = String.format("%s/DataWWW2019/UserEmbedding/%s_%s_embedding_dim_%d_fold_%d.txt", prefix, data, model, dim, fold);
                    if(model.equals("BOW"))
                        embedFile = String.format("%s/DataWWW2019/UserEmbedding/%s_%s.txt", prefix, data, model);
                    String testInterFile = String.format("./data/DataEUB/CV4Edges/%sCVIndex4Interaction_fold_%d_test.txt", data, fold);
                    String testNonInterFile = String.format("./data/DataEUB/CV4Edges/%sCVIndex4NonInteraction_time_%d_fold_%d.txt", data, time, fold);

                    if(model.equals("BOW"))
                        link = new  LinkPredictionWithUserEmbeddingBOW()  ;
                    else link = new LinkPredictionWithUserEmbedding();

                    link.initLinkPred(idFile, embedFile, testInterFile, testNonInterFile);
                    link.calculateAllNDCGMAP();
                    perfs[fold][t] = link.calculateAvgNDCGMAP();
                }
            }
            allFoldsPerf.put(model, perfs);
        }
        link.calcMeanStd(allFoldsPerf);
    }
}
