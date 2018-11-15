package Application.LinkPrediction4EUB;

import Application.LinkPrediction4MMB.LinkPrediction;
import utils.Utils;

import java.io.*;
import java.util.*;

/**
 * @author Lin Gong (lg5bt@virginia.edu)
 */
public class LinkPredictionWithUserEmbedding {

    class _User4Link {
        String m_uid;
        int m_index;
        double[] m_embedding;
        ArrayList<_Edge4Link> m_edges;

        public _User4Link(String uid, int idx, double[] embedding){
            m_uid = uid;
            m_index = idx;
            m_embedding = embedding;
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

    double[][] m_similarity;
    double[][] m_embeddings;
    ArrayList<String> m_testUserIds;
    HashMap<String, Integer> m_idIndexMap;
    HashMap<String, _User4Link> m_testUserMap;
    protected Object m_NDCGMAPLock = null;

    public LinkPredictionWithUserEmbedding(){
        m_idIndexMap = new HashMap<>();
        m_testUserMap = new HashMap<>();
        m_NDCGMAPLock = new Object();
        m_testUserIds = new ArrayList<>();
    }

    // load each user's embedding
    public void loadUserEmbedding(String filename){
        try {
            File file = new File(filename);
            BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(file), "UTF-8"));
            String line;

            String firstLine = reader.readLine();
            String[] strs = firstLine.trim().split("\t");
            if(strs.length != 2){
                System.out.println("[error]The file is not correct!! Double check user embedding file!");
                return;
            }
            m_userSize = Integer.valueOf(strs[0]);
            m_dim = Integer.valueOf(strs[1]);
            m_embeddings = new double[m_userSize][m_dim];

            // read each user's embedding one by one
            int count = 0;
            while ((line = reader.readLine()) != null) {
                if(count > m_userSize){
                    System.out.println("[error]The line number exceeds the user size!!");
                    break;
                }
                String[] valStrs = line.trim().split("\t");
                if(valStrs.length != m_dim + 1){
                    System.out.println("[error]The user's dimension is not correct!!");
                    continue;
                }
                String uid = valStrs[0];
                double[] embedding = new double[m_dim];
                for(int i=1; i<valStrs.length; i++){
                    embedding[i-1] = Double.valueOf(valStrs[i]);
                }
                m_idIndexMap.put(uid, count);
                m_embeddings[count++] = embedding;
            }
            reader.close();
            System.out.format("[Info]Finish loading %d user embeddings from %s.\n", count, filename);
        } catch(IOException e){
            e.printStackTrace();
        }
    }

    // load edges (interactions + non-interactions) for all the users
    public void loadTestOneEdges(String filename){
        int label = 1, count = 0;
        try {
            File file = new File(filename);
            BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(file), "UTF-8"));
            String line;
            while ((line = reader.readLine()) != null) {
                String[] strs = line.trim().split("\t");
                String uid = strs[0];
                int index = m_idIndexMap.get(uid);
                _User4Link ui = new _User4Link(strs[0], index, m_embeddings[index]);
                m_testUserMap.put(uid, ui);
                m_testUserIds.add(uid);
                for(int j=1; j<strs.length; j++){
                    String ujId = strs[j];
                    int ujIdx = m_idIndexMap.get(ujId);
                    double sim = getSimilarity(ui.getIndex(), ujIdx);
                    ui.addOneEdge(new _Edge4Link(ujIdx, sim, label));
                }
                count++;
            }
            reader.close();
            System.out.format("[Info]%d/%d users' interactions are loaded!!\n", m_testUserMap.size(), count);
        } catch(IOException e){
            e.printStackTrace();
        }
    }

    // load edges (interactions + non-interactions) for all the users
    public void loadTestZeroEdges(String filename){
        int label = 0, count = 0;
        try {
            File file = new File(filename);
            BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(file), "UTF-8"));
            String line;
            while ((line = reader.readLine()) != null) {
                String[] strs = line.trim().split("\t");
                String uid = strs[0];
                if(!m_testUserMap.containsKey(uid))
                    continue;
                _User4Link ui = m_testUserMap.get(uid);
                for(int j=1; j<strs.length; j++){
                    String ujId = strs[j];
                    int ujIdx = m_idIndexMap.get(ujId);
                    double sim = getSimilarity(ui.getIndex(), ujIdx);
                    ui.addOneEdge(new _Edge4Link(ujIdx, sim, label));
                }
                count++;
            }
            reader.close();
            System.out.format("[Info]%d/%d users' non-interactions are loaded!!\n", m_testUserMap.size(), count);
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
        m_similarity = new double[m_userSize][m_userSize];
        for(int i=0; i<m_userSize; i++){
            for(int j=i+1; j<m_userSize; j++){
                m_similarity[i][j] = Utils.cosine(m_embeddings[i], m_embeddings[j]);
            }
        }
    }

    public void ininLinkPred(String embedFile, String testInterFile, String testNonInterFile) {
        loadUserEmbedding(embedFile);
        calcSimilarity();
        // load both one-edges ans zero-edges
        loadTestOneEdges(testInterFile);
        loadTestZeroEdges(testNonInterFile);
    }

    // The function for calculating all NDCGs and MAPs.
    public void calculateAllNDCGMAP(){
        m_NDCGs = new double[m_testUserIds.size()];
        m_MAPs = new double[m_testUserIds.size()];
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

                        for (int i = 0; i + core <m_testUserIds.size(); i += numOfCores) {
                            if(i%500==0) System.out.print(".");
                            String uid = m_testUserIds.get(i + core);
                            _User4Link user = m_testUserMap.get(uid);
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
    public double[] calculateNDCGMAP(_User4Link user){

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
//		System.out.format("DCG:%.2f, IDCG:%.2f\n", DCG, iDCG);
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

    public static void main(String[] args){
        String data = "StackOverflow";
        for(int dim: new int[]{10}) {

            for (int fold : new int[]{1}) {
                int[] times = new int[]{2, 3, 4};
                String[] models = new String[]{"EUB", "LDA", "HFT"}; // "LDA", "HFT", "TADW""

                String prefix = "";
                double[][][] perfs = new double[models.length][times.length][2];
                for (int t = 0; t < times.length; t++) {
                    int time = times[t];
                    for (int i = 0; i < models.length; i++) {
                        String model = models[i];
                        System.out.format("-----current model-%s-time-%d-dim-%d------\n", model, time, dim);

                        String embedFile = String.format("/Users/lin/DataWWW2019/UserEmbedding/%s_%s_embedding_dim_%d_fold_%d.txt", data, model, dim, fold);
                        String testInterFile = String.format("./data/DataEUB/CV4Edges/%sCVIndex4Interaction_fold_%d_test.txt", data, fold);
                        String testNonInterFile = String.format("./data/DataEUB/CV4Edges/%sCVIndex4NonInteraction_time_%d_fold_%d.txt", data, time, fold);

                        LinkPredictionWithUserEmbedding link = new LinkPredictionWithUserEmbedding();
                        link.ininLinkPred(embedFile, testInterFile, testNonInterFile);
                        link.calculateAllNDCGMAP();
                        perfs[i][t] = link.calculateAvgNDCGMAP();

                    }
                }
                for (int time : times) {
                    System.out.format("\t\t%d\t\t", time);
                }
                System.out.println();
                for (int i = 0; i < models.length; i++) {
                    System.out.print(models[i] + "\t");
                    for (double[] ndcgMap : perfs[i]) {
                        System.out.format("%.4f\t%.4f\t", ndcgMap[0], ndcgMap[1]);
                    }
                    System.out.println();
                }
            }
        }
    }
}
