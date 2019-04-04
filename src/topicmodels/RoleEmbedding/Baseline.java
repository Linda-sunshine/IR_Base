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

    private double m_converge, m_alpha, m_beta, m_stepSize; //parameter for regularization
    private int m_dim, m_nuOfRoles, m_numberOfIteration;
    private double[][] m_users; // U*M
    private double[][] m_roles; // L*M, i.e., B in the derivation
    private ArrayList<String> m_uIds;
    private HashMap<String, Integer> m_uId2IndexMap;
    private HashMap<Integer, ArrayList<Integer>> m_oneEdges;
    private HashMap<Integer, ArrayList<Integer>> m_zeroEdges;

    public Baseline(int m, int L, int nuIter, double converge, double alpha, double beta, double stepSize){
        m_dim = m;
        m_nuOfRoles = L;

        m_numberOfIteration = nuIter;
        m_converge = converge;
        m_alpha = alpha;
        m_beta = beta;
        m_stepSize = stepSize;

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
            System.out.format("[Info]Finish loading user ids from %s\n", filename);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    // load connections/nonconnections from files
    public void loadOneEdges(String filename){
//        HashMap<Integer, ArrayList<Integer>> edgeMap = eij == 1 ? m_oneEdges : m_zeroEdges;
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
                if(!m_oneEdges.containsKey(uiId)){
                    m_oneEdges.put(uiIdx, new ArrayList<>());
                }
                m_oneEdges.get(uiIdx).add(ujIdx);
                count++;
            }
            System.out.format("[Info]Finish loading one edges of %d users' %d links from %s\n", m_oneEdges.size(), count);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void sampleZeroEdges(){

    }

    public void init(){
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
    }

    public void train(){

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

        double affinity, gTermOne, fValue;
        double lastFValue = 1.0, converge = 1e-6, diff, iterMax = 10, iter = 0;
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
                    fValue += sigmod(affinity);
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
                    fValue += sigmod(-affinity);
                    gTermOne = sigmod(affinity);
                    // each dimension of user vectors ui and uj
                    for(int g=0; g<m_dim; g++){
                        userG[uiIdx][g] += gTermOne * calcUserGradientTermTwo(g, m_users[ujIdx]);
                        userG[ujIdx][g] += gTermOne * calcUserGradientTermTwo(g, m_users[uiIdx]);
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
                    m_users[i][j] -= m_stepSize * userG[i][j];
                }
            }
            diff = (lastFValue - fValue) / lastFValue;
            lastFValue = fValue;
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

    // update role vectors;
    public double updateRoleVectors(){

        double fValue, lastFValue = 1.0, converge = 1e-6, diff, iterMax = 10, iter = 0;
        double[][] gTermTwo = new double[m_roles.length][m_dim];
        double[][] roleG = new double[m_roles.length][m_dim];

        do {
            fValue = 0;
            for (double[] g : gTermTwo) {
                Arrays.fill(g, 0);
            }
            // updates of gradient from one edges
            for (int uiIdx : m_oneEdges.keySet()) {
                for(int ujIdx: m_oneEdges.get(uiIdx)){
                    fValue += calcRoleGradientWithOneEdge(gTermTwo, uiIdx, ujIdx, m_users[uiIdx], m_users[ujIdx]);
                }
            }
            // updates of gradient from zero edges
            for (int uiIdx : m_zeroEdges.keySet()) {
                for(int ujIdx: m_zeroEdges.get(uiIdx)){
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
            // update the role vectors based on the gradients
            for(int l=0; l<m_roles.length; l++){
                for(int m=0; m<m_dim; m++){
                    m_roles[l][m] -= m_stepSize * roleG[l][m];
                }
            }
            diff = fValue - lastFValue;
            lastFValue = fValue;
        } while(iter++ < iterMax && Math.abs(diff) > converge);
        return fValue;
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
        double coef = 2/sigmod(-affinity); // 2/(1+exp(u_i^T B^T B u_j))
        for(int m=0; m<ui.length; m++){
            for(int n=0; n<uj.length; n++){
                g[m][n] += coef * ui[m] * uj[n];
            }
        }
        return Math.log(sigmod(affinity));
    }

    // directly add the update to the gradient of role
    public double calcRoleGradientWithZeroEdge(double[][] g, int i, int j, double[] ui, double[] uj){
        double affinity = calcAffinity(i, j);
        double coef = 2/sigmod(affinity); // 2/(1+exp(u_i^T B^T B u_j))
        for(int m=0; m<ui.length; m++){
            for(int n=0; n<uj.length; n++){
                g[m][n] -= coef * ui[m] * uj[n];
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
                for(int m=1; m<m_dim; m++){
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

    //The main function for general link pred
    public static void main(String[] args){

    }
}
