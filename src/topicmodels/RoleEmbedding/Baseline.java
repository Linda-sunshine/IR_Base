package topicmodels.RoleEmbedding;

import structures._User4EUB;
import utils.Utils;

import java.io.*;
import java.util.*;
import java.util.HashMap;

/**
 * @author Lin Gong (lg5bt@virginia.edu)
 * The joint modeling of  role embedding (L*M) and user embedding (U*M)
 * The class contains the simple solution of directly designed objective function.
 */

public class Baseline {

    boolean m_updateRoleByMatrix = false;
    private double m_converge, m_alpha, m_beta, m_stepSize; //parameter for regularization
    private int m_dim, m_nuOfRoles, m_numberOfIteration;
    private double[][] m_users; // U*M
    private double[][] m_roles; // L*M, i.e., B in the derivation
    private ArrayList<String> m_uIds;
    private HashMap<String, Integer> m_uId2IndexMap;
    private HashMap<Integer, HashSet<Integer>> m_oneEdges;
    private HashMap<Integer, HashSet<Integer>> m_zeroEdges;

    public Baseline(int m, int L, int nuIter, double converge, double alpha, double beta, double stepSize){
        m_dim = m;
        m_nuOfRoles = L;

        m_numberOfIteration = nuIter;
        m_converge = converge;
        m_alpha = alpha;
        m_beta = beta;
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
                ujId = strs[1];
                uiIdx = m_uId2IndexMap.get(uiId);
                ujIdx = m_uId2IndexMap.get(ujId);
                if(!m_uId2IndexMap.containsKey(uiId)){
                    System.out.println("The user does not exist in the user set!");
                    continue;
                }
                if(!edgeMap.containsKey(uiIdx)){
                    edgeMap.put(uiIdx, new HashSet<>());
                }
                edgeMap.get(uiIdx).add(ujIdx);
                count++;
                if(count % 10000 == 0)
                    System.out.print(".");
                if(count % 1000000 == 0)
                    System.out.println();
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
        m_roles = new double[m_nuOfRoles][m_dim];

        for(double[] user: m_users){
            initOneVector(user);
        }
        for(double[] role: m_roles){
            initOneVector(role);
        }
    }

    // initialize each user vector
    public void initOneVector(double[] vct){
        for(int i=0; i<vct.length; i++){
            vct[i] = Math.random();
        }
        Utils.normalize(vct);
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

            // update role vectors;
            currentFunctionValue += updateRoleVectors();

            if (iter++ > 0)
                converge = Math.abs((lastFunctionValue - currentFunctionValue) / lastFunctionValue);
            else
                converge = 1.0;

            lastFunctionValue = currentFunctionValue;

        } while (iter < m_numberOfIteration && converge > m_converge);
    }

    // update user vectors;
    public double updateUserVectors(){

        System.out.println("Start optimizing user vectors...");
        double affinity, gTermOne, fValue;
        double lastFValue = 1.0, converge = 1e-6, diff, iterMax = 3, iter = 0;
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
                    affinity = calcAffinity(uiIdx, ujIdx);
                    fValue += Math.log(sigmod(affinity));
                    gTermOne = sigmod(-affinity);
                    // each dimension of user vectors ui and uj
                    for(int g=0; g<m_dim; g++){
                        userG[uiIdx][g] += gTermOne * calcUserGradientTermTwo(g, m_users[ujIdx]);
                        userG[ujIdx][g] += gTermOne * calcUserGradientTermTwo(g, m_users[uiIdx]);
                    }
                }
            }
            // updates based on zero edges
            for(int uiIdx: m_zeroEdges.keySet()){
                for(int ujIdx: m_zeroEdges.get(uiIdx)){
                    if(ujIdx <= uiIdx) continue;
                    // for each edge
                    affinity = calcAffinity(uiIdx, ujIdx);
                    fValue += Math.log(sigmod(-affinity));
                    gTermOne = sigmod(affinity);
                    // each dimension of user vectors ui and uj
                    for(int g=0; g<m_dim; g++){
                        userG[uiIdx][g] -= gTermOne * calcUserGradientTermTwo(g, m_users[ujIdx]);
                        userG[ujIdx][g] -= gTermOne * calcUserGradientTermTwo(g, m_users[uiIdx]);
                    }
                }
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

    // calculate the second part of the gradient for user vector
    public double calcUserGradientTermTwo(int g, double[] uj){
        double val = 0;
        for(int p=0; p<m_dim; p++){
            for(int l=0; l<m_nuOfRoles; l++){
                val += m_roles[l][g] * m_roles[l][p] * uj[p];
            }
        }
        return val;
    }

    public double updateRoleVectors(){
        if(m_updateRoleByMatrix)
            return updateRoleVectorsByMatrix();
        else
            return updateRoleVectorsByElement();
    }

    // update role vectors;
    public double updateRoleVectorsByMatrix(){

        System.out.println("Start optimizing role vectors...");
        double fValue, lastFValue = 1.0, converge = 1e-6, diff, iterMax = 5, iter = 0;
        double[][] gTermTwo = new double[m_dim][m_dim];
        double[][] roleG = new double[m_roles.length][m_dim];

        do {
            fValue = 0;
            for (double[] g : gTermTwo) {
                Arrays.fill(g, 0);
            }
            // updates of gradient from one edges
            for (int uiIdx : m_oneEdges.keySet()) {
                for(int ujIdx: m_oneEdges.get(uiIdx)){
                    if(ujIdx <= uiIdx) continue;
                    fValue += calcRoleGradientWithOneEdge(gTermTwo, uiIdx, ujIdx, m_users[uiIdx], m_users[ujIdx]);
                }
            }
            // updates of gradient from zero edges
            for (int uiIdx : m_zeroEdges.keySet()) {
                for(int ujIdx: m_zeroEdges.get(uiIdx)){
                    if(ujIdx <= uiIdx) continue;
                    fValue += calcRoleGradientWithZeroEdge(gTermTwo, uiIdx, ujIdx, m_users[uiIdx], m_users[ujIdx]);
                }
            }
            // multiply: B * roleG - 2*beta*B_{gh}
            for(int l=0; l<m_nuOfRoles; l++){
                for(int m=0; m<m_dim; m++){
                    roleG[l][m] = Utils.dotProduct(m_roles[l], getOneColumn(gTermTwo, m)) - 2 * m_beta * m_roles[l][m];
                    fValue -= m_beta * m_roles[l][m] * m_roles[l][m];
                }
            }
            System.out.format("Function value: %.1f\n", fValue);
            // update the role vectors based on the gradients
            for(int l=0; l<m_roles.length; l++){
                for(int m=0; m<m_dim; m++){
                    m_roles[l][m] += m_stepSize * 0.01 * roleG[l][m];
                }
            }
            diff = fValue - lastFValue;
            lastFValue = fValue;
        } while(iter++ < iterMax && Math.abs(diff) > converge);
        return fValue;
    }

    // update role vectors;
    public double updateRoleVectorsByElement(){

        System.out.println("Start optimizing role vectors...");
        double fValue, fValueOne, fValueZero, affinity, gTermOne, lastFValue = 1.0, converge = 1e-6, diff, iterMax = 5, iter = 0;
        double[][] roleG = new double[m_roles.length][m_dim];

        do {
            fValueOne = 0; fValueZero = 0;
            for (double[] g : roleG) {
                Arrays.fill(g, 0);
            }
            // updates of gradient from one edges
            for (int uiIdx : m_oneEdges.keySet()) {
                for(int ujIdx: m_oneEdges.get(uiIdx)){
                    if(ujIdx <= uiIdx) continue;

                    affinity = calcAffinity(uiIdx, ujIdx);
                    fValueOne += Math.log(sigmod(affinity));
                    gTermOne = sigmod(-affinity);
                    // each element of role embedding B_{gh}
                    for(int g=0; g<m_nuOfRoles; g++){
                        for(int h=0; h<m_dim; h++){
                            roleG[g][h] += gTermOne * calcRoleGradientTermTwo(g, h, m_users[uiIdx], m_users[ujIdx]);
                        }
                    }
                }
            }
            // updates of gradient from zero edges
            for (int uiIdx : m_zeroEdges.keySet()) {
                for(int ujIdx: m_zeroEdges.get(uiIdx)){
                    if(ujIdx <= uiIdx) continue;
                    affinity = calcAffinity(uiIdx, ujIdx);
                    fValueZero += Math.log(sigmod(-affinity));
                    gTermOne = sigmod(affinity);
                    // each element of role embedding B_{gh}
                    for(int g=0; g<m_nuOfRoles; g++){
                        for(int h=0; h<m_dim; h++){
                            roleG[g][h] -= gTermOne * calcRoleGradientTermTwo(g, h, m_users[uiIdx], m_users[ujIdx]);
                        }
                    }
                }
            }
            fValue = fValueOne + fValueZero;
            for(int l=0; l<m_nuOfRoles; l++){
                for(int m=0; m<m_dim; m++){
                    roleG[l][m] -= 2 * m_beta * m_roles[l][m];
                    fValue -= m_beta * m_roles[l][m] * m_roles[l][m];
                }
            }

            // update the role vectors based on the gradients
            for(int l=0; l<m_roles.length; l++){
                for(int m=0; m<m_dim; m++){
                    m_roles[l][m] += m_stepSize * 0.01 * roleG[l][m];
                }
            }
            diff = fValue - lastFValue;
            lastFValue = fValue;
            System.out.format("Function value: %.1f\n", fValue);
        } while(iter++ < iterMax && Math.abs(diff) > converge);
        return fValue;
    }

    // calculate the second part of the gradient for user vector
    public double calcRoleGradientTermTwo(int g, int h, double[] ui, double[] uj){
        double val = 0;
        for(int p=0; p<m_dim; p++){
            val += ui[h] * m_roles[g][p] * uj[p];
            val += ui[p] * m_roles[g][p] * uj[h];
        }
        return val;
    }

    public double[] getOneColumn(double[][] mtx, int j){
        double[] col = new double[mtx.length];
        for(int i=0; i<mtx.length; i++){
            col[i] = mtx[i][j];
        }
        return col;
    }

    // directly add the update to the gradient of role
    public double calcRoleGradientWithOneEdge(double[][] g, int i, int j, double[] ui, double[] uj){
        double affinity = calcAffinity(i, j);
        double coef = sigmod(-affinity); // 1/(1+exp(u_i^T B^T B u_j))
        for(int m=0; m<ui.length; m++){
            for(int n=0; n<uj.length; n++){
                g[m][n] += coef * (ui[m] * uj[n] + uj[m] * ui[n]);
            }
        }
        return Math.log(sigmod(affinity));
    }

    // directly add the update to the gradient of role
    public double calcRoleGradientWithZeroEdge(double[][] g, int i, int j, double[] ui, double[] uj){
        double affinity = calcAffinity(i, j);
        double coef = sigmod(affinity); // 2/(1+exp(u_i^T B^T B u_j))
        for(int m=0; m<ui.length; m++){
            for(int n=0; n<uj.length; n++){
                g[m][n] -= coef * (ui[m] * uj[n] + uj[m] * ui[n]);
            }
        }
        return Math.log(sigmod(-affinity));
    }

    // u_i^T B^T B u_j
    public double calcAffinity(int i, int j){
        double res = 0;
        double[] ui = m_users[i];
        double[] uj = m_users[j];
        for(int p=0; p<m_dim; p++){
            for(int l=0; l<m_nuOfRoles; l++){
                for(int m=0; m<m_dim; m++){
                    res += ui[m] * m_roles[l][m] * m_roles[l][p] * uj[p];
                }
            }
        }
        return res;
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

    public void printRoleEmbedding(String filename) {
        try {
            PrintWriter writer = new PrintWriter(new File(filename));
            writer.format("%d\t%d\n", m_roles.length, m_dim);
            for (int i = 0; i < m_roles.length; i++) {
                writer.format("%d\t", i);
                for (double v : m_roles[i]) {
                    writer.format("%.4f\t", v);
                }
                writer.write("\n");
            }
            writer.close();
            System.out.format("Finish writing %d role embeddings!", m_roles.length);
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

    public void printUids(String filename, HashSet<String> uids){
        try{
            PrintWriter writer = new PrintWriter(new File(filename));
            for(String uid: uids){
                writer.write(uid+"\n");
            }
            writer.close();
        } catch(IOException e){
            e.printStackTrace();
        }
    }
    //The main function for general link pred
    public static void main(String[] args){

        String dataset = "yelp"; // "release-youtube"

        String userFile = String.format("./data/RoleEmbedding/%s-users.txt", dataset);
        String oneEdgeFile = String.format("./data/RoleEmbedding/%s-links.txt", dataset);
        String zeroEdgeFile = String.format("./data/RoleEmbedding/%s-nonlinks.txt", dataset);
        String userEmbeddingFile = String.format("/home/lin/DataWWW2019/UserEmbedding/YelpNew_Role_embedding_dim_10_fold_0.txt", dataset);
        String roleEmbeddingFile = String.format("/home/lin/DataWWW2019/UserEmbedding/YelpNew_role_embedding.txt", dataset);

        int m = 20, L = 30, nuIter = 100;
        double converge = 1e-6, alpha = 0.5, beta = 0.5, stepSize = 0.001;
        Baseline base = new Baseline(m, L, nuIter, converge, alpha, beta, stepSize);

        base.loadUsers(userFile);
        base.loadEdges(oneEdgeFile, 1); // load one edges

//        base.sampleZeroEdges();
//        base.saveZeroEdges(zeroEdgeFile);

        base.loadEdges(zeroEdgeFile, 0); // load zero edges
        base.train();
        base.printUserEmbedding(userEmbeddingFile);
        base.printRoleEmbedding(roleEmbeddingFile);
    }
}
