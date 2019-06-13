package topicmodels.RoleEmbedding;

import java.util.Arrays;

public class RoleEmbeddingNonNegative extends RoleEmbeddingBaseline {

    // the vector can take any value, m_usersInput is the softmax(m_usersInputPrime)
    double[][] m_usersInputPrime;
    double[][] m_inputPrimeG;

    public RoleEmbeddingNonNegative(int m, int L, int nuIter, double converge, double alpha, double beta, double stepSize){
        super(m, L, nuIter, converge, alpha, beta, stepSize);
    }

    @Override
    public String toString() {
        return String.format("MultiRoleEmbedding_NonNegative[dim:%d, #Roles:%d, alpha:%.4f, beta:%.4f, #Iter:%d]", m_dim,
                m_nuOfRoles, m_alpha, m_beta, m_numberOfIteration);
    }


    public void init(){
        m_usersInputPrime = new double[m_uIds.size()][m_dim];
        m_usersInput = new double[m_uIds.size()][m_dim];

        m_inputPrimeG = new double[m_uIds.size()][m_dim];

        for(int i=0; i<m_usersInputPrime.length; i++){
            double[] userPrime = m_usersInputPrime[i];
            // init the original vector
            initOneVector(userPrime);
            m_usersInput[i] = softmax(userPrime);
        }

        // update the role vectors
        m_roles = new double[m_nuOfRoles][m_dim];
        m_rolesG = new double[m_nuOfRoles][m_dim];
        for(double[] role: m_roles){
            initOneVector(role);
        }
    }

    public double[] softmax(double[] arr){
        double[] res = new double[arr.length];
        // calculate the sum of the exp of the array
        for(int i=0; i<arr.length; i++) {
            res[i] = Math.exp(arr[i]);
        }
        double sum = utils.Utils.sumOfArray(res);
        for(int i=0; i<arr.length; i++) {
            res[i] /= sum;
        }
        return res;
    }

    // update user vectors with non-negative constrains
    public double updateUserVectors(){

        System.out.println("Start optimizing user vectors...");
        double affinity, gTermOne, fValue;
        double lastFValue = 1.0, converge = 1e-6, diff, iterMax = 3, iter = 0;
        double[] ui, uj, uiPrime, ujPrime;

        do{
            fValue = 0;
            for(double[] g: m_inputPrimeG){
                Arrays.fill(g, 0);
            }
            // updates based on one edges
            for(int uiIdx: m_oneEdges.keySet()){
                for(int ujIdx: m_oneEdges.get(uiIdx)){
                    if(ujIdx <= uiIdx) continue;
                    // for each edge
                    ui = m_usersInput[uiIdx];
                    uj = m_usersInput[ujIdx];
                    uiPrime = m_usersInputPrime[uiIdx];
                    ujPrime = m_usersInputPrime[ujIdx];
                    affinity = calcAffinity(uiIdx, ujIdx);
                    fValue -= Math.log(sigmod(affinity));
                    gTermOne = sigmod(-affinity);
                    // each dimension of user vectors ui and uj
                    for(int g=0; g<m_dim; g++){
                        m_inputPrimeG[uiIdx][g] -= gTermOne * calcUserGradientTermTwo(g, uj) * calcUserGradientTermThree(g, uiPrime);
                        m_inputPrimeG[ujIdx][g] -= gTermOne * calcUserGradientTermTwo(g, ui) * calcUserGradientTermThree(g, ujPrime);
                    }
                }
            }
            // updates based on zero edges
            if(m_zeroEdges.size() != 0){
                fValue += updateUserVectorsWithSampledZeroEdges();
            }

            // add the gradient from regularization
            for(int i=0; i<m_usersInputPrime.length; i++){
                for(int m=0; m<m_dim; m++){
                    // the default one is L2 regularization
                    if(!m_L1){
                        m_inputPrimeG[i][m] += m_alpha * 2 * m_usersInputPrime[i][m];
                    }
                }
            }
            // update the user vectors based on the gradients
            for(int i=0; i<m_usersInputPrime.length; i++){
                for(int j=0; j<m_dim; j++){
                    if(m_L1){
                        fValue += m_alpha * Math.abs(m_usersInputPrime[i][j]);
                        m_usersInputPrime[i][j] = calcProx( m_usersInputPrime[i][j] - m_stepSize * m_inputPrimeG[i][j]);

                    } else{
                        fValue += m_alpha * m_usersInputPrime[i][j] * m_usersInputPrime[i][j];
                        m_usersInputPrime[i][j] -= m_stepSize * m_inputPrimeG[i][j];

                    }
                }
            }

            // update the user vector based on userPrime
            for(int i=0; i<m_usersInput.length; i++){
                m_usersInput[i] = softmax(m_usersInputPrime[i]);
            }

            diff = (lastFValue - fValue) / lastFValue;
            lastFValue = fValue;
            System.out.format("Function value: %.1f\n", fValue);
        } while(iter++ < iterMax && Math.abs(diff) > converge);
        return fValue;
    }


    // if sampled zero edges are load, user sampled zero edges for update
    public double updateUserVectorsWithSampledZeroEdges(){
        double fValue = 0, affinity, gTermOne, ui[], uj[], uiPrime[], ujPrime[];
        for(int uiIdx: m_zeroEdges.keySet()){
            for(int ujIdx: m_zeroEdges.get(uiIdx)){
                if(ujIdx <= uiIdx) continue;
                // for each edge
                ui = m_usersInput[uiIdx];
                uj = m_usersInput[ujIdx];
                uiPrime = m_usersInputPrime[uiIdx];
                ujPrime = m_usersInputPrime[ujIdx];
                affinity = calcAffinity(uiIdx, ujIdx);
                fValue -= Math.log(sigmod(-affinity));
                gTermOne = sigmod(affinity);
                // each dimension of user vectors ui and uj
                for(int g=0; g<m_dim; g++){
                    m_inputPrimeG[uiIdx][g] += gTermOne * calcUserGradientTermTwo(g, uj) * calcUserGradientTermThree(g, uiPrime);
                    m_inputPrimeG[ujIdx][g] += gTermOne * calcUserGradientTermTwo(g, ui) * calcUserGradientTermThree(g, ujPrime);
                }
            }
        }
        return fValue;
    }


    // calculate the third part of the gradient for user vector, the gradient of ui regarding to ui'
    public double calcUserGradientTermThree(int g, double[] uiPrime){
        double sum = 0;
        double expUig = Math.exp(uiPrime[g]);
        for(int p=0; p<m_dim; p++){
            sum += Math.exp(uiPrime[p]);
        }
        return expUig * (sum - expUig) / (sum * sum);
    }
//
//    // update role vectors;
//    public double updateRoleVectorsByElement(){
//
//        System.out.println("Start optimizing role vectors...");
//        double fValue, affinity, gTermOne, lastFValue = 1.0, converge = 1e-6, diff, iterMax = 5, iter = 0;
//
//        do {
//            fValue = 0;
//            for (double[] g : m_rolesG) {
//                Arrays.fill(g, 0);
//            }
//            // updates of gradient from one edges
//            for (int uiIdx : m_oneEdges.keySet()) {
//                for(int ujIdx: m_oneEdges.get(uiIdx)){
//                    if(ujIdx <= uiIdx) continue;
//
//                    affinity = calcAffinity(uiIdx, ujIdx);
//                    fValue -= Math.log(sigmod(affinity));
//                    gTermOne = sigmod(-affinity);
//                    // each element of role embedding B_{gh}
//                    for(int g=0; g<m_nuOfRoles; g++){
//                        for(int h=0; h<m_dim; h++){
//                            m_rolesG[g][h] -= gTermOne * calcRoleGradientTermTwo(g, h, m_usersInput[uiIdx], m_usersInput[ujIdx]);
//                        }
//                    }
//                }
//            }
//
//            // updates based on zero edges
//            if(m_zeroEdges.size() != 0){
//                fValue += updateRoleVectorsWithSampledZeroEdgesByElement();
//            }
//
//            for(int l=0; l<m_nuOfRoles; l++){
//                for(int m=0; m<m_dim; m++){
//                    m_rolesG[l][m] += 2 * m_beta * m_roles[l][m];
//                    fValue += m_beta * m_roles[l][m] * m_roles[l][m];
//
//                }
//            }
//
//            // update the role vectors based on the gradients
//            for(int l=0; l<m_roles.length; l++){
//                for(int m=0; m<m_dim; m++){
//                    m_roles[l][m] -= m_stepSize * m_rolesG[l][m];
//                }
//            }
//            diff = fValue - lastFValue;
//            lastFValue = fValue;
//            System.out.format("Function value: %.1f\n", fValue);
//        } while(iter++ < iterMax && Math.abs(diff) > converge);
//        return fValue;
//    }

    //The main function for general link pred
    public static void main(String[] args){

        String dataset = "YelpNew", model = "multirole_non_neg_l1";
        boolean l1 = true;

        int fold = 0, dim = 10, nuOfRoles = 10, nuIter = 50, order = 1;
        double converge = 1e-6, alpha = 0.01, beta = 1, stepSize = 0.1;

        String userFile = String.format("./data/RoleEmbedding/%s_userids.txt", dataset);
        String oneEdgeFile = String.format("./data/RoleEmbedding/%sCVIndex4Interaction_fold_%d_train.txt", dataset, fold);
        String zeroEdgeFile = String.format("./data/RoleEmbedding/%sCVIndex4NonInteractions_fold_%d_train_2.txt", dataset, fold);
        String oneEdgeTestFile = String.format("./data/RoleEmbedding/%sCVIndex4Interaction_fold_%d_test.txt", dataset, fold);
        String zeroEdgeTestFile = String.format("./data/RoleEmbedding/%sCVIndex4NonInteractions_fold_%d_test.txt", dataset, fold);

        String userEmbeddingFile = String.format("/Users/lin/DataWWW2019/UserEmbedding%d/%s_%s_l2_embedding_alpha_%.4f_step_size_%.4f_iter_%d_order_%d_nuOfRoles_%d_dim_%d_fold_%d.txt", order, dataset, model, alpha, stepSize, nuIter, order, nuOfRoles, dim, fold);
        String roleEmbeddingFile = String.format("/Users/lin/DataWWW2019/UserEmbedding%d/%s_role_l2_embedding_alpha_%.4f_step_size_%.4f_iter_%d_order_%d_nuOfRoles_%d_dim_%d_fold_%d.txt", order, dataset, alpha, stepSize, nuIter, order, nuOfRoles, dim, fold);

        RoleEmbeddingNonNegative roleNonNeg = new RoleEmbeddingNonNegative(dim, nuOfRoles, nuIter, converge, alpha, beta, stepSize);

        roleNonNeg.loadUsers(userFile);
        if(order >= 1)
            roleNonNeg.loadEdges(oneEdgeFile, 1);
        if(order >= 2)
            roleNonNeg.generate2ndConnections();
        if(order >= 3)
            roleNonNeg.generate3rdConnections();

        roleNonNeg.loadEdges(zeroEdgeFile, 0); // load zero edges
//        roleBase.loadEdges(oneEdgeTestFile, -1); // one edges for testing
//        roleBase.loadEdges(zeroEdgeTestFile, -2);


        roleNonNeg.setL1Regularization(l1);

        if(l1){
            userEmbeddingFile = String.format("/Users/lin/DataWWW2019/UserEmbedding%d/%s_%s_embedding_alpha_%.4f_step_size_%.4f_iter_%d_order_%d_nuOfRoles_%d_dim_%d_fold_%d.txt", order, dataset, model, alpha, stepSize, nuIter, order, nuOfRoles, dim, fold);
            roleEmbeddingFile = String.format("/Users/lin/DataWWW2019/UserEmbedding%d/%s_%s_role_embedding_alpha_%.4f_step_size_%.4f_iter_%d_order_%d_nuOfRoles_%d_dim_%d_fold_%d.txt", order, dataset, model, alpha, stepSize, nuIter, order, nuOfRoles, dim, fold);
        }

        roleNonNeg.train();
        roleNonNeg.printUserEmbedding(userEmbeddingFile);
        roleNonNeg.printRoleEmbedding(roleEmbeddingFile, roleNonNeg.getRoleEmbeddings());
    }
}
