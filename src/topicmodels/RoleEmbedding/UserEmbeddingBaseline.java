package topicmodels.RoleEmbedding;

import utils.Utils;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;

public class UserEmbeddingBaseline {

    protected double m_converge, m_alpha, m_stepSize; //parameter for regularization
    protected int m_dim, m_numberOfIteration;
    protected double[][] m_users; // U*M

    protected ArrayList<String> m_uIds;
    protected HashMap<String, Integer> m_uId2IndexMap;
    protected HashMap<Integer, HashSet<Integer>> m_oneEdges;
    protected HashMap<Integer, HashSet<Integer>> m_zeroEdges;


    public UserEmbeddingBaseline(int m, int nuIter, double converge, double alpha, double stepSize){
        m_dim = m;
        m_numberOfIteration = nuIter;
        m_converge = converge;
        m_alpha = alpha;
        m_stepSize = stepSize;

        m_uIds = new ArrayList<>();
        m_uId2IndexMap = new HashMap<>();
        m_oneEdges = new HashMap<>();
        m_zeroEdges = new HashMap<>();
    }

    // load user ids from file
    public void loadUsers(String filename){
        try {
            // load beta for the whole corpus first
            File userFile = new File(filename);
            BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(userFile),
                    "UTF-8"));
            String line;
            int count = 0;
            while ((line = reader.readLine()) != null){
                // start reading one user's id
                String uid = line.trim();
                m_uIds.add(uid);
                m_uId2IndexMap.put(uid, count++);
            }
            System.out.format("[Info]Finish loading %d user ids from %s\n", m_uIds.size(), filename);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    // load connections/nonconnections from files
    public void loadEdges(String filename, int eij){
        HashMap<Integer, HashSet<Integer>> edgeMap = eij == 1 ? m_oneEdges : m_zeroEdges;
        try {
            // load beta for the whole corpus first
            File linkFile = new File(filename);
            BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(linkFile),
                    "UTF-8"));
            String line, uiId, ujId, strs[];
            int uiIdx, ujIdx, count = 0;
            while ((line = reader.readLine()) != null){
                // start reading one user's id
                strs = line.trim().split("\\s+");
                if(strs.length < 2){
                    System.out.println("Invalid pair!");
                    continue;
                }
                uiId = strs[0];
                uiIdx = m_uId2IndexMap.get(uiId);

                for(int j=1; j<strs.length; j++) {
                    ujId = strs[j];
                    ujIdx = m_uId2IndexMap.get(ujId);

                    if (!m_uId2IndexMap.containsKey(uiId)) {
                        System.out.println("The user does not exist in the user set!");
                        continue;
                    }
                    if (!edgeMap.containsKey(uiIdx)) {
                        edgeMap.put(uiIdx, new HashSet<>());
                    }
                    edgeMap.get(uiIdx).add(ujIdx);
                    count++;
                    if (count % 10000 == 0)
                        System.out.print(".");
                    if (count % 1000000 == 0)
                        System.out.println();
                }
            }
            System.out.format("\n[Info]Finish loading %d edges of %d users' %d links from %s\n", eij, edgeMap.size(),
                    count, filename);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void sampleZeroEdges() {
        for(int i=0; i<m_uIds.size(); i++){
            if(i % 10000 == 0)
                System.out.print(".");
            if(i % 1000000 == 0)
                System.out.println();
            String uiId = m_uIds.get(i);
            int uiIdx = m_uId2IndexMap.get(uiId);
            if(!m_zeroEdges.containsKey(uiIdx)){
                m_zeroEdges.put(uiIdx, new HashSet<>());
            }
            HashSet<Integer> zeroEdges = m_zeroEdges.get(uiIdx);
            HashSet<Integer> oneEdges = m_oneEdges.containsKey(uiIdx) ? m_oneEdges.get(uiIdx) : null;
            int number = m_oneEdges.containsKey(uiIdx) ? m_oneEdges.get(uiIdx).size() : 1;

            while(zeroEdges.size() < number) {
                String ujId = m_uIds.get((int) (Math.random() * m_uIds.size()));
                int ujIdx = m_uId2IndexMap.get(ujId);
                if (oneEdges == null || !oneEdges.contains(ujIdx)) {
                    zeroEdges.add(ujIdx);
                    if(!m_zeroEdges.containsKey(ujIdx))
                        m_zeroEdges.put(ujIdx, new HashSet<>());
                    m_zeroEdges.get(ujIdx).add(uiIdx);
                }
            }

        }
    }
    public void saveZeroEdges(String filename){

        try{
            int count = 0;
            PrintWriter writer = new PrintWriter(new File(filename));
            for(int uiIdx: m_zeroEdges.keySet()){
                String uiId = m_uIds.get(uiIdx);
                HashSet<Integer> zeroEdges = m_zeroEdges.get(uiIdx);
                for(int ujIdx: zeroEdges){
                    writer.format("%s\t%s\n", uiId, m_uIds.get(ujIdx));
                    count++;
                }
            }
            writer.close();
            System.out.format("Finish writing %d zero edges.\n", count);
        } catch(IOException e){
            e.printStackTrace();
        }
    }

    public void init(){
        m_users = new double[m_uIds.size()][m_dim];
        for(double[] user: m_users){
            initOneVector(user);
        }
    }

    // initialize each user vector
    public void initOneVector(double[] vct){
        for(int i=0; i<vct.length; i++){
            vct[i] = Math.random();
        }
        Utils.normalize(vct);
    }

    // update user vectors;
    public double updateUserVectors(){

        System.out.println("Start optimizing user vectors...");
        double affinity, gTermOne, fValue;
        double lastFValue = 1.0, converge = 1e-6, diff, iterMax = 3, iter = 0;
        double[] ui, uj;
        double[][] userG = new double[m_users.length][m_dim];

        do{
            fValue = 0;
            for(double[] g: userG){
                Arrays.fill(g, 0);
            }
            // updates based on one edges
            for(int uiIdx: m_oneEdges.keySet()){
                for(int ujIdx: m_oneEdges.get(uiIdx)){
                    if(ujIdx <= uiIdx) continue;
                    // for each edge
                    ui = m_users[uiIdx];
                    uj = m_users[ujIdx];
                    affinity = Utils.dotProduct(ui, uj);
                    fValue += Math.log(sigmod(affinity));
                    gTermOne = sigmod(-affinity);
                    // each dimension of user vectors ui and uj
                    for(int g=0; g<m_dim; g++){
                        userG[uiIdx][g] += gTermOne * uj[g];
                        userG[ujIdx][g] += gTermOne * ui[g];
                    }
                }
            }
            // updates based on zero edges
            if(m_zeroEdges == null){
                fValue += updateUserVectorsWithAllZeroEdges(userG);

            } else{
                fValue += updateUserVectorsWithSampledZeroEdges(userG);
            }

            // add the gradient from regularization
            for(int i=0; i<m_users.length; i++){
                for(int m=0; m<m_dim; m++){
                    userG[i][m] -= m_alpha * 2 * m_users[i][m];
                }
            }
            // update the user vectors based on the gradients
            for(int i=0; i<m_users.length; i++){
                for(int j=0; j<m_dim; j++){
                    fValue -= m_alpha * m_users[i][j] * m_users[i][j];
                    m_users[i][j] += m_stepSize * userG[i][j];
                }
            }
            diff = (lastFValue - fValue) / lastFValue;
            lastFValue = fValue;
            System.out.format("Function value: %.1f\n", fValue);
        } while(iter++ < iterMax && Math.abs(diff) > converge);
        return fValue;
    }

    // if no zero edges are loaded, user all zero edges for udpate
    public double updateUserVectorsWithAllZeroEdges(double[][] userG){
        double fValue = 0, gTermOne, affinity, ui[], uj[];
        for(int i=0; i<m_uIds.size(); i++){
            ui = m_users[i];
            // collect zero edges first
            for(int j=i+1; j<m_uIds.size(); j++){
                if(m_oneEdges.containsKey(i) && m_oneEdges.get(i).contains(j)) continue;
                uj = m_users[j];
                affinity = Utils.dotProduct(ui, uj);
                fValue += Math.log(sigmod(-affinity));
                gTermOne = sigmod(affinity);
                // each dimension of user vectors ui and uj
                for(int g=0; g<m_dim; g++){
                    userG[i][g] -= gTermOne * calcUserGradientTermTwo(g, uj);
                    userG[j][g] -= gTermOne * calcUserGradientTermTwo(g, ui);
                }
            }
        }
        return fValue;
    }

    // if sampled zero edges are load, user sampled zero edges for update
    public double updateUserVectorsWithSampledZeroEdges(double[][] userG){
        double fValue = 0, affinity, gTermOne, ui[], uj[];
        for(int uiIdx: m_zeroEdges.keySet()){
            for(int ujIdx: m_zeroEdges.get(uiIdx)){
                if(ujIdx <= uiIdx) continue;
                // for each edge
                ui = m_users[uiIdx];
                uj = m_users[ujIdx];
                affinity = Utils.dotProduct(ui, uj);
                fValue += Math.log(sigmod(-affinity));
                gTermOne = sigmod(affinity);
                // each dimension of user vectors ui and uj
                for(int g=0; g<m_dim; g++){
                    userG[uiIdx][g] -= gTermOne * calcUserGradientTermTwo(g, uj);
                    userG[ujIdx][g] -= gTermOne * calcUserGradientTermTwo(g, ui);
                }
            }
        }
        return fValue;
    }

    public double calcUserGradientTermTwo(int g, double[] uj){
        return uj[g];
    }

    public void train(){

        init();
        int iter = 0;
        double lastFunctionValue = -1.0;
        double currentFunctionValue;
        double converge;
        // iteratively update user vectors and role vectors until converge
        do {
            System.out.format(String.format("\n----------Start EM %d iteraction----------\n", iter));

            // update user vectors;
            currentFunctionValue = updateUserVectors();

            if (iter++ > 0)
                converge = Math.abs((lastFunctionValue - currentFunctionValue) / lastFunctionValue);
            else
                converge = 1.0;

            lastFunctionValue = currentFunctionValue;

        } while (iter < m_numberOfIteration && converge > m_converge);
    }


    public double[] getOneColumn(double[][] mtx, int j){
        double[] col = new double[mtx.length];
        for(int i=0; i<mtx.length; i++){
            col[i] = mtx[i][j];
        }
        return col;
    }

    public double sigmod(double v){
        return 1/(1 + Math.exp(-v));
    }

    // print out learned user embedding
    public void printUserEmbedding(String filename) {
        try {
            PrintWriter writer = new PrintWriter(new File(filename));
            writer.format("%d\t%d\n", m_users.length, m_dim);
            for (int i = 0; i < m_users.length; i++) {
                writer.format("%s\t", m_uIds.get(i));
                for (double v : m_users[i]) {
                    writer.format("%.4f\t", v);
                }
                writer.write("\n");
            }
            writer.close();
            System.out.format("Finish writing %d user embeddings!", m_users.length);

        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void preprocessYelpData(String filename, String uidFilename, String linkFilename){
        HashSet<String> uids = new HashSet<>();
        HashSet<String> ujds = new HashSet<>();
        int nuOfLinks = 0;
        try {
            // load beta for the whole corpus first
            File linkFile = new File(filename);

            BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(linkFile),
                    "UTF-8"));
            PrintWriter idWriter = new PrintWriter(new File(uidFilename));
            PrintWriter linkWriter = new PrintWriter(new File(linkFilename));
            String line, strs[];
            while ((line = reader.readLine()) != null) {
                // start reading one user's id
                strs = line.trim().split("\\s+");
                String uid = strs[0];
                idWriter.write(uid + "\n");
                for (int i = 1; i < strs.length; i++) {
                    nuOfLinks++;
                    linkWriter.format("%s\t%s\n", uid, strs[i]);
                }
            }
            idWriter.close();
            linkWriter.close();
            System.out.format("\n[Info]Number of links: %d\n", nuOfLinks);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

//    public void printUids(String filename, HashSet<String> uids){
//        try{
//            PrintWriter writer = new PrintWriter(new File(filename));
//            for(String uid: uids){
//                writer.write(uid+"\n");
//            }
//            writer.close();
//        } catch(IOException e){
//            e.printStackTrace();
//        }
//    }

    //The main function for general link pred
    public static void main(String[] args){

        String dataset = "YelpNew"; // "release-youtube"

        int fold = 0, m = 10, nuIter = 200;
        String userFile = String.format("./data/RoleEmbedding/%sUserIds.txt", dataset);
        String oneEdgeFile = String.format("./data/RoleEmbedding/%sCVIndex4Interaction_fold_%d_train.txt", dataset, fold);
        String zeroEdgeFile = String.format("./data/RoleEmbedding/%sCVIndex4NonInteractions_fold_%d_train.txt", dataset, fold);
        String userEmbeddingFile = String.format("/Users/lin/DataWWW2019/UserEmbedding/%s_User_embedding_dim_%d_fold_%d.txt", dataset, m, fold);

        double converge = 1e-6, alpha = 0.5, stepSize = 0.01;
        UserEmbeddingBaseline base = new UserEmbeddingBaseline(m, nuIter, converge, alpha, stepSize);

        base.loadUsers(userFile);
        base.loadEdges(oneEdgeFile, 1); // load one edges
//        base.loadEdges(zeroEdgeFile, 0); // load zero edges

//        base.sampleZeroEdges();
//        base.saveZeroEdges(zeroEdgeFile);

        base.train();
        base.printUserEmbedding(userEmbeddingFile);
    }
}
