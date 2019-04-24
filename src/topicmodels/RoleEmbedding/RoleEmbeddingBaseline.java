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

public class RoleEmbeddingBaseline extends UserEmbeddingBaseline{

    protected boolean m_updateRoleByMatrix = false;
    protected double m_beta; //parameter for regularization
    protected int m_nuOfRoles;
    protected double[][] m_roles, m_rolesG; // L*M, i.e., B in the derivation


    public RoleEmbeddingBaseline(int m, int L, int nuIter, double converge, double alpha, double beta, double stepSize){
        super(m, nuIter, converge, alpha, stepSize);
        m_nuOfRoles = L;
        m_beta = beta;
    }

    @Override
    public String toString() {
        return String.format("MultiRoleEmbedding_Baseline[dim:%d, #Roles:%d, alpha:%.4f, beta:%.4f, #Iter:%d]", m_dim,
                m_nuOfRoles, m_alpha, m_beta, m_numberOfIteration);
    }

    public void init(){
        super.init();
        m_roles = new double[m_nuOfRoles][m_dim];
        m_rolesG = new double[m_nuOfRoles][m_dim];
        for(double[] role: m_roles){
            initOneVector(role);
        }
    }

    @Override
    public void train(){
        System.out.println(toString());

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

    @Override
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

        do {
            fValue = 0;
            // we don't need to clear m_roleG as its value is directly replaced
            for (double[] g : gTermTwo) {
                Arrays.fill(g, 0);
            }
            // updates of gradient from one edges
            for (int uiIdx : m_oneEdges.keySet()) {
                for(int ujIdx: m_oneEdges.get(uiIdx)){
                    if(ujIdx <= uiIdx) continue;
                    fValue += calcRoleGradientWithOneEdge(gTermTwo, uiIdx, ujIdx, m_usersInput[uiIdx], m_usersInput[ujIdx]);
                }
            }

            // updates based on zero edges
            if(m_zeroEdges.size() != 0){
                fValue += updateRoleVectorsWithSampledZeroEdgesByMatrix(gTermTwo);
            }

            // multiply: B * roleG - 2*beta*B_{gh}
            for(int l=0; l<m_nuOfRoles; l++){
                for(int m=0; m<m_dim; m++){
                    m_rolesG[l][m] = Utils.dotProduct(m_roles[l], getOneColumn(gTermTwo, m)) - 2 * m_beta * m_roles[l][m];
                    fValue -= m_beta * m_roles[l][m] * m_roles[l][m];
                }
            }
            System.out.format("Function value: %.1f\n", fValue);
            // update the role vectors based on the gradients
            for(int l=0; l<m_roles.length; l++){
                for(int m=0; m<m_dim; m++){
                    m_roles[l][m] += m_stepSize * 0.01 * m_rolesG[l][m];
                }
            }
            diff = fValue - lastFValue;
            lastFValue = fValue;
        } while(iter++ < iterMax && Math.abs(diff) > converge);
        return fValue;
    }

//    public double updateRoleVectorsWithAllZeroEdgesByMatrix(double[][] gTermTwo){
//        // updates of gradient from zero edges
//        double fValueZero = 0;
//        for(int i=0; i<m_uIds.size(); i++){
//            for(int j=i+1; j<m_uIds.size(); j++){
//                if(m_oneEdges.containsKey(i) && m_oneEdges.get(i).contains(j)) continue;
//                fValueZero += calcRoleGradientWithZeroEdge(gTermTwo, i, j, m_usersInput[i], m_usersInput[j]);
//            }
//        }
//        return fValueZero;
//    }

    public double updateRoleVectorsWithSampledZeroEdgesByMatrix(double[][] gTermTwo){
        // updates of gradient from zero edges
        double fValueZero = 0;
        for (int uiIdx : m_zeroEdges.keySet()) {
            for(int ujIdx: m_zeroEdges.get(uiIdx)){
                if(ujIdx <= uiIdx) continue;
                fValueZero += calcRoleGradientWithZeroEdge(gTermTwo, uiIdx, ujIdx, m_usersInput[uiIdx], m_usersInput[ujIdx]);
            }
        }
        return fValueZero;
    }

    // update role vectors;
    public double updateRoleVectorsByElement(){

        System.out.println("Start optimizing role vectors...");
        double fValue, affinity, gTermOne, lastFValue = 1.0, converge = 1e-6, diff, iterMax = 5, iter = 0;

        do {
            fValue = 0;
            for (double[] g : m_rolesG) {
                Arrays.fill(g, 0);
            }
            // updates of gradient from one edges
            for (int uiIdx : m_oneEdges.keySet()) {
                for(int ujIdx: m_oneEdges.get(uiIdx)){
                    if(ujIdx <= uiIdx) continue;

                    affinity = calcAffinity(uiIdx, ujIdx);
                    fValue += Math.log(sigmod(affinity));
                    gTermOne = sigmod(-affinity);
                    // each element of role embedding B_{gh}
                    for(int g=0; g<m_nuOfRoles; g++){
                        for(int h=0; h<m_dim; h++){
                            m_rolesG[g][h] += gTermOne * calcRoleGradientTermTwo(g, h, m_usersInput[uiIdx], m_usersInput[ujIdx]);
                        }
                    }
                }
            }

            // updates based on zero edges
            if(m_zeroEdges.size() != 0){
                fValue += updateRoleVectorsWithSampledZeroEdgesByElement();
            }

            for(int l=0; l<m_nuOfRoles; l++){
                for(int m=0; m<m_dim; m++){
                    m_rolesG[l][m] -= 2 * m_beta * m_roles[l][m];
                    fValue -= m_beta * m_roles[l][m] * m_roles[l][m];
                }
            }

            // update the role vectors based on the gradients
            for(int l=0; l<m_roles.length; l++){
                for(int m=0; m<m_dim; m++){
                    m_roles[l][m] += m_stepSize * 0.01 * m_rolesG[l][m];
                }
            }
            diff = fValue - lastFValue;
            lastFValue = fValue;
            System.out.format("Function value: %.1f\n", fValue);
        } while(iter++ < iterMax && Math.abs(diff) > converge);
        return fValue;
    }

//    public double updateRoleVectorsWithAllZeroEdgesByElement(){
//        double affinity, gTermOne, fValueZero = 0;
//        for(int i=0; i<m_uIds.size(); i++){
//            for(int j=i+1; j<m_uIds.size(); j++){
//                if(m_oneEdges.containsKey(i) && m_oneEdges.get(i).contains(j)) continue;
//                affinity = calcAffinity(i, j);
//                fValueZero += Math.log(sigmod(-affinity));
//                gTermOne = sigmod(affinity);
//                // each element of role embedding B_{gh}
//                for(int g=0; g<m_nuOfRoles; g++){
//                    for(int h=0; h<m_dim; h++){
//                        m_rolesG[g][h] -= gTermOne * calcRoleGradientTermTwo(g, h, m_usersInput[i], m_usersInput[j]);
//                    }
//                }
//            }
//        }
//        return fValueZero;
//
//    }

    public double updateRoleVectorsWithSampledZeroEdgesByElement(){
        double affinity, gTermOne, fValueZero = 0;
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
                        m_rolesG[g][h] -= gTermOne * calcRoleGradientTermTwo(g, h, m_usersInput[uiIdx], m_usersInput[ujIdx]);
                    }
                }
            }
        }
        return fValueZero;
    }

//    // if no zero edges are loaded, user all zero edges for udpate
//    public double updateRoleVectorsWithAllZeroEdges(){
//        double fValue = 0, gTermOne, affinity, ui[], uj[];
//        for(int i=0; i<m_uIds.size(); i++){
//            ui = m_usersInput[i];
//            // collect zero edges first
//            for(int j=i+1; j<m_uIds.size(); j++){
//                if(m_oneEdges.containsKey(i) && m_oneEdges.get(i).contains(j)) continue;
//                uj = m_usersInput[j];
//                affinity = calcAffinity(i, j);
//                fValue += Math.log(sigmod(-affinity));
//                gTermOne = sigmod(affinity);
//                // each dimension of user vectors ui and uj
//                for(int g=0; g<m_nuOfRoles; g++){
//                    for(int h=0; h<m_dim; h++){
//                        m_rolesG[g][h] -= gTermOne * calcRoleGradientTermTwo(g, h, m_usersInput[i], m_usersInput[j]);
//                    }
//                }
//            }
//        }
//        return fValue;
//    }

    // calculate the second part of the gradient for user vector
    public double calcRoleGradientTermTwo(int g, int h, double[] ui, double[] uj){
        double val = 0;
        for(int p=0; p<m_dim; p++){
            val += ui[h] * m_roles[g][p] * uj[p];
            val += ui[p] * m_roles[g][p] * uj[h];
        }
        return val;
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
    @Override
    public double calcAffinity(int i, int j){

        double res = 0;
        double[] ui = m_usersInput[i];
        double[] uj = m_usersInput[j];
        for(int p=0; p<m_dim; p++){
            for(int l=0; l<m_nuOfRoles; l++){
                for(int m=0; m<m_dim; m++){
                    res += ui[m] * m_roles[l][m] * m_roles[l][p] * uj[p];
                }
            }
        }
        return res;
    }

    public void printRoleEmbedding(String filename, double[][] roles) {
        try {
            PrintWriter writer = new PrintWriter(new File(filename));
            writer.format("%d\t%d\n", roles.length, m_dim);
            for (int i = 0; i < roles.length; i++) {
                writer.format("%d\t", i);
                for (double v : roles[i]) {
                    writer.format("%.4f\t", v);
                }
                writer.write("\n");
            }
            writer.close();
            System.out.format("Finish writing %d role embeddings!", roles.length);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
    public double[][] getRoleEmbeddings(){
        return m_roles;
    }
    //The main function for general link pred
    public static void main(String[] args){

        String dataset = "FB"; // "release-youtube"
        int fold = 0, dim = 10, nuOfRoles = 10, nuIter = 100;

        String userFile = String.format("./data/RoleEmbedding/%sUserIds.txt", dataset);
        String oneEdgeFile = String.format("./data/RoleEmbedding/%sCVIndex4Interaction_fold_%d_train.txt", dataset, fold);
        String zeroEdgeFile = String.format("./data/RoleEmbedding/%sCVIndex4NonInteractions_fold_%d_train_2.txt", dataset, fold);
        String userEmbeddingFile = String.format("/Users/lin/DataWWW2019/UserEmbedding/%s_multirole_embedding_#roles_%d_dim_%d_fold_%d.txt", dataset, nuOfRoles, dim, fold);
        String roleEmbeddingFile = String.format("/Users/lin/DataWWW2019/UserEmbedding/%s_role_embedding_#roles_%d_dim_%d_fold_%d.txt", dataset, nuOfRoles, dim, fold);

        double converge = 1e-6, alpha = 1, beta = 1, stepSize = 0.001;
        RoleEmbeddingBaseline roleBase = new RoleEmbeddingBaseline(dim, nuOfRoles, nuIter, converge, alpha, beta, stepSize);

        roleBase.loadUsers(userFile);
        roleBase.loadEdges(oneEdgeFile, 1); // load one edges
        roleBase.generate2ndConnections();

//        roleBase.sampleZeroEdges();
//        roleBase.saveZeroEdges(zeroEdgeFile);

        roleBase.loadEdges(zeroEdgeFile, 0); // load zero edges
        roleBase.train();
        roleBase.printUserEmbedding(userEmbeddingFile);
        roleBase.printRoleEmbedding(roleEmbeddingFile, roleBase.getRoleEmbeddings());
    }
}
