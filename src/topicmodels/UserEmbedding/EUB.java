package topicmodels.UserEmbedding;

import Jama.Matrix;
import LBFGS.LBFGS;
import structures.*;
import topicmodels.LDA.LDA_Variational;
import utils.Utils;

import java.util.*;

/**
 * @author Lin Gong (lg5bt@virginia.edu)
 * The joint modeling of user embedding (U*M) and topic embedding (K*M)
 */

public class EUB extends LDA_Variational {

    protected ArrayList<_Topic4EUB> m_topics;
    protected ArrayList<_User4EUB> m_users;
    protected ArrayList<_Doc4EUB> m_docs;

    protected int m_embedding_dim;

    // assume this is the look-up table for doc-user pairs
    protected HashMap<String, Integer> m_usersIndex;
    // key: user index, value: document index array
    protected HashMap<Integer, ArrayList<Integer>> m_userDocMap;
    // assume i follows js, key: index i, value: index js
    protected HashMap<Integer, HashSet<Integer>> m_userI2JMap;
    // assume i does not follow j', key: index i, value: index j'
    protected HashMap<Integer, HashSet<Integer>> m_userI2JPrimeMap;

    // assume ps follows i, key: index i, value: index ps
    protected HashMap<Integer, HashSet<Integer>> m_userP2IMap;
    // assume p' does not follow i, key: index p', value: index i
    protected HashMap<Integer, HashSet<Integer>> m_userPPrime2IMap;

    // key: doc index, value: user index
    protected HashMap<Integer, Integer> m_docUserMap;

    /*****variational parameters*****/
    protected double t_mu = 1.0, t_sigma = 1.0;
    protected double d_mu = 1.0, d_sigma = 1.0;
    protected double u_mu = 1.0, u_sigma = 1.0;

    /*****model parameters*****/
    // this alpha is differnet from alpha in LDA
    // alpha is precision parameter for topic embedding in EUB
    // alpha is a vector parameter for dirichlet distribution
    protected double m_alpha_s;
    protected double m_tau;
    protected double m_gamma;
    protected double m_xi;

    public EUB(int number_of_iteration, double converge, double beta,
               _Corpus c, double lambda, int number_of_topics, double alpha,
               int varMaxIter, double varConverge, int m) {
        super(number_of_iteration, converge, beta, c, lambda, number_of_topics, alpha,
                varMaxIter, varConverge);
        m_embedding_dim = m;
        m_topics = new ArrayList<>();
        m_users = new ArrayList<>();
        m_docs = new ArrayList<>();
    }

    // Load the data for later user
    protected void buildLookupTables(ArrayList<_User> users){
        // build the lookup table for user/doc/topic
        for(int i=0; i< users.size(); i++){
            _User4EUB user = (_User4EUB) users.get(i);
            m_users.add(user);
            m_usersIndex.put(user.getUserID(), i);
            m_userDocMap.put(i, new ArrayList<>());
            for(_Doc d: user.getReviews()){
                _Doc4EUB doc = (_Doc4EUB) d;
                int docIndex = m_docs.size();
                doc.setID(docIndex);
                m_docs.add(doc);
                m_userDocMap.get(i).add(docIndex);
                m_docUserMap.put(docIndex, i);
            }
        }

        for(int k=0; k<number_of_topics; k++){
            m_topics.add(new _Topic4EUB(k));
        }
    }

    // build the network: interactions and non-interactions
    protected double buildNetwork(){

    }


    @Override
    public void EMonCorpus(){

        m_trainSet = new ArrayList<>();
        // collect all training reviews
        for(_User u: m_users){
            for(_Review r: u.getTrainReviews()){
                m_trainSet.add(r);
            }
        }
        EM();
    }

    @Override
    public void EM() {
        initialize_probability(m_trainSet);

        int iter = 0;
        double lastAllLikelihood = 1.0;
        double currentAllLikelihood;
        double converge;
        do {
            currentAllLikelihood = E_step();

            if (iter >= 0)
                converge = Math.abs((lastAllLikelihood - currentAllLikelihood) / lastAllLikelihood);
            else
                converge = 1.0;

            calculate_M_step(iter);

            lastAllLikelihood = currentAllLikelihood;
        } while( ++iter < number_of_iteration && converge > m_converge);
    }

    @Override
    /***to be done***/
    protected void init() { // clear up for next iteration during EM
        super.init();
    }

    @Override
    protected void initialize_probability(Collection<_Doc> docs) {

        System.out.println("[Info]Initializing topics, documents and users...");

        for(_Topic4EUB t: m_topics)
            t.setTopics4Variational(m_embedding_dim, t_mu, t_sigma);

        for(_Doc d: m_trainSet)
            ((_Doc4EUB) d).setTopics4Variational(number_of_topics, d_alpha, d_mu, d_sigma);

        for(_User4EUB u: m_users)
            u.setTopics4Variational(m_embedding_dim, m_users.size(), u_mu, u_sigma);

        init();

        for(_Doc doc: m_trainSet)
            updateStats4Doc((_Doc4EUB) doc);

        calculate_M_step(0);
    }

    // Update the stats related with doc
    protected void updateStats4Doc(_Doc4EUB doc){

        _SparseFeature[] fv = doc.getSparse();
        int wid;
        double v;
        for(int n=0; n<fv.length; n++) {
            wid = fv[n].getIndex();
            v = fv[n].getValue();
            for(int i=0; i<number_of_topics; i++)
                word_topic_sstat[i][wid] += v * doc.m_phi[n][i];
        }
    }

    // update variational parameters of latent variables
    protected double E_step(){
        int iter = 0;
        double totalLikelihood, last = -1.0, converge;

        init();
        do {
            totalLikelihood = 0.0;
            for (_Doc doc: m_trainSet) {
                totalLikelihood += varInference4Doc((_Doc4EUB) doc);
            }
            for (_Topic4EUB topic: m_topics)
                totalLikelihood += varInference4Topic(topic);

            for(_User4EUB user: m_users)
                totalLikelihood += varInference4User(user);

            if(Double.isNaN(totalLikelihood) || Double.isInfinite(totalLikelihood))
                System.out.println("[error] The likelihood is Nan or Infinity!!");

            if(iter > 0)
                converge = Math.abs((totalLikelihood - last) / last);
            else
                converge = 1.0;

            last = totalLikelihood;

            if(iter % 10 == 0)
                System.out.format("[Info]Single-thread E-Step: %d iteration, likelihood=%.2f, converge to %.8f\n",
                        iter, last, converge);

        }while(iter++ < m_varMaxIter && converge > m_varConverge);

        for(_Doc doc: m_trainSet)
            updateStats4Doc((_Doc4EUB) doc);

        return totalLikelihood;
    }

    @Override
    public void calculate_M_step(int iter){

        est_alpha();// precision for topic embedding
        est_gamma(); // precision for user embedding
        est_beta(); // topic-word distribution
        est_tau(); // precision for topic proportion
        est_xi(); // sigma for the affinity \delta_{ij}
    }

    protected void est_alpha(){
        double denominator = 0;
        for(int k=0; k<number_of_topics; k++){
            _Topic4EUB topic = m_topics.get(k);
            denominator += sumSigmaDiagAddMuTransposeMu(topic.m_sigma_phi, topic.m_mu_phi);
        }
        m_alpha_s = denominator!=0 ? (number_of_topics * m_embedding_dim / denominator) : 0;
    }

    protected void est_gamma(){
        double denominator = 0;
        for(int uIndex: m_userDocMap.keySet()){
            _User4EUB user = m_users.get(uIndex);
            denominator += sumSigmaDiagAddMuTransposeMu(user.m_sigma_u, user.m_mu_u);
        }
        m_gamma = denominator!=0 ? (m_users.size() * m_embedding_dim) / denominator : 0;
    }

    protected void est_beta(){
        for(int k=0; k<number_of_topics; k++) {
            double sum = Utils.sumOfArray(word_topic_sstat[k]);
            for(int v=0; v<vocabulary_size; v++) //will be in the log scale!!
                topic_term_probabilty[k][v] = Math.log(word_topic_sstat[k][v]/sum);
        }
    }

    protected void est_tau(){
        double denominator = 0;
        double D = 0; // total number of documents
        for(int uIndex: m_userDocMap.keySet()){
            _User4EUB user = m_users.get(uIndex);
            for(int dIndex: m_userDocMap.get(uIndex)){
                _Doc4EUB doc = m_docs.get(dIndex);
                denominator += calculateStat4Tau(user, doc);
            }
        }
        m_tau = denominator != 0 ? D * number_of_topics / denominator : 0;

    }


    protected void est_xi(){
        double xiSquare = 0, term1 = 0, term2 = 0, term3 = 0;
        // sum over all interactions and non-interactions
        // traverse four edge maps instead due to sparsity
        // i->j and j'
        for(int i: m_userI2JMap.keySet()){
            _User4EUB ui = m_users.get(i);
            // js
            HashSet<Integer> js = m_userI2JMap.get(i);
            HashSet<Integer> jPrimes = m_userI2JPrimeMap.get(i);
            for(int j: js){
                _User4EUB uj = m_users.get(j);
                term1 = ui.m_mu_delta[j] * ui.m_mu_delta[j] + ui.m_sigma_delta[j] * ui.m_sigma_delta[j];
                for(int m=0; m<m_embedding_dim; m++){
                    term2 =
                }
            }

        }

        for(int p: m_userP2IMap.keySet()){

        }

    }

    protected double calculateStat4Xi(_User4EUB ui, _User4EUB uj){

    }

    protected double calculateStat4Tau(_User4EUB user, _Doc4EUB doc){
        double term1 = 0, term2 = 0, term3 = 0;

        for(int k=0; k<number_of_topics; k++){
            _Topic4EUB topic = m_topics.get(k);
            term1 += doc.m_sigma_theta[k] + doc.m_mu_theta[k] * doc.m_mu_theta[k];
            for(int m=0 ; m<m_embedding_dim; m++){
                term2 += 2 * doc.m_mu_theta[k] * topic.m_mu_phi[m] * user.m_mu_u[m];
                for(int l=0; l<m_embedding_dim; l++){
                    term3 += (user.m_sigma_u[m][l] + user.m_mu_u[m] * user.m_mu_u[l])
                            * (topic.m_sigma_phi[m][l] + topic.m_mu_phi[m] * topic.m_mu_phi[l]);
                }
            }
        }
        return term1 + term2 + term3;
    }

    protected double varInference4Topic(_Topic4EUB topic){
        // update the mu and sigma for each topic \phi_k -- Eq(65) and Eq(67)
        update_phi_k(topic);

        return calc_log_likelihood_per_topic(topic);
    }

    protected double varInference4User(_User4EUB user){
        // update the variational parameters for user embedding \u_i -- Eq(70) and Eq(72)
        update_u_i(user);
        // update the mean and variance for pair-wise affinity \mu^{delta_{ij}}, \sigma^{\delta_{ij}} -- Eq(75)
        update_delta_ij_mu(user);
        update_delta_ij_sigma(user);
        // update the taylor parameter epsilon
        update_epsilon(user);

        return calc_log_likelihood_per_user(user);
    }

    protected double varInference4Doc(_Doc4EUB doc){

        // variational parameters for word indicator z_{idn}
        update_eta_id(doc);
        // variational parameters for topic distribution \theta_{id}
        update_theta_id_mu(doc);
        update_theta_id_sigma(doc);
        // taylor parameter
        update_zeta(doc);

        return calc_log_likelihood_per_doc(doc);
    }

    // update the mu and sigma for each topic \phi_k -- E(65) and Eq(67)
    protected void update_phi_k(_Topic4EUB topic){

        Matrix term1 = new Matrix(new double[m_embedding_dim][m_embedding_dim]);
        double[] term2 = new double[m_embedding_dim];

        // \tau * \sum_u|sum_d(\sigma + \mu * \mu^T)
        for(int uIndex: m_userDocMap.keySet()){
            _User4EUB user = m_users.get(uIndex);
            // \sigma + mu * mu^T
            for(int dIndex: m_userDocMap.get(uIndex)){
                _Doc4EUB doc = m_docs.get(dIndex);
                Utils.add2Array(term2, user.m_mu_u, m_tau * doc.m_mu_theta[topic.getIndex()]);
            }
            int docSize = m_userDocMap.get(uIndex).size();
            Matrix tmp = new Matrix(sigmaAddMuMuTranspose(user.m_sigma_u, user.m_mu_u));
            term1.plusEquals(tmp.timesEquals(docSize));
        }
        // * \tau
        term1.timesEquals(m_tau);
        double[][] diag = new double[m_embedding_dim][m_embedding_dim];
        for(int i=0; i<m_embedding_dim; i++){
            diag[i][i] = m_alpha_s;
        }
        // + \alpha * I
        term1.plusEquals(new Matrix(diag));
        Matrix invsMtx = term1.inverse();

        topic.m_sigma_phi = invsMtx.getArray();
        topic.m_mu_phi = invsMtx.times(new Matrix(term2, 1)).getArray()[0];
    }

    // \sum_{mm}\simga[m][m] + \mu^T * \mu
    protected double sumSigmaDiagAddMuTransposeMu(double[][] sigma, double[] mu){
        if(sigma.length != mu.length)
            return 0;
        int dim = mu.length;
        double sum = 0;
        for(int m=0; m<dim; m++){
            sum += sigma[m][m] + mu[m] * mu[m];
        }
        return sum;
    }

    // \simga + \mu * \mu^T
    protected double[][] sigmaAddMuMuTranspose(double[][] sigma, double[] mu){
        int dim = mu.length;
        double[][] res = new double[dim][dim];
        for(int i=0; i<dim; i++){
            for(int j=0; j<dim; j++){
                res[i][j] = sigma[i][j] + mu[i]*mu[j];
            }
        }
        return res;
    }

    // update the variational parameters for user embedding \u_i -- Eq(70) and Eq(72)
    protected void update_u_i(_User4EUB user){
        // term1: the current user's document
        Matrix sigma_term1 = new Matrix(new double[m_embedding_dim][m_embedding_dim]);
        Matrix sigma_term2 = new Matrix(new double[m_embedding_dim][m_embedding_dim]);
        Matrix sigma_term3 = new Matrix(new double[m_embedding_dim][m_embedding_dim]);

        double[] mu_term1 = new double[m_embedding_dim];
        double[] mu_term2 = new double[m_embedding_dim];
        double[] mu_term3 = new double[m_embedding_dim];

        int i = m_usersIndex.get(user.getUserID());
        int docSize = m_userDocMap.get(i).size();
        for(int k=0; k<number_of_topics; k++){
            // stat for updating sigma
            _Topic4EUB topic = m_topics.get(k);
            Matrix tmp = new Matrix(sigmaAddMuMuTranspose(topic.m_sigma_phi, topic.m_mu_phi));
            sigma_term1.plusEquals(tmp.timesEquals(docSize));

            // stat for updating mu
            for(int dIndex: m_userDocMap.get(i)){
                _Doc4EUB doc = m_docs.get(dIndex);
                Utils.add2Array(mu_term1, topic.m_mu_phi, m_tau * doc.m_mu_theta[topic.getIndex()]);
            }
        }
        // * \tau
        sigma_term1.timesEquals(m_tau);
        // \gamma * I
        double[][] diag = new double[m_embedding_dim][m_embedding_dim];
        for(int a=0; a<m_embedding_dim; a++){
            diag[a][a] = m_gamma;
        }
        // + \gamma * I
        sigma_term1.plusEquals(new Matrix(diag));

        // term2: user_i -> user_j
        for(int j: m_userI2JMap.get(i)){
            _User4EUB uj = m_users.get(j);
            double delta_ij = user.m_mu_delta[j];
            sigma_term2.plusEquals(new Matrix(sigmaAddMuMuTranspose(uj.m_sigma_u, uj.m_mu_u)));
            Utils.add2Array(mu_term2, uj.m_mu_u, delta_ij/m_xi/m_xi);
        }
        sigma_term2.timesEquals(1/m_xi /m_xi);

        // term3: user_p -> user_i
        for(int p: m_userP2IMap.get(i)){
            _User4EUB up = m_users.get(p);
            double delta_pi = up.m_mu_delta[i];
            sigma_term3.plusEquals(new Matrix(sigmaAddMuMuTranspose(up.m_sigma_u, up.m_mu_u)));
            Utils.add2Array(mu_term3, up.m_mu_u, delta_pi/m_xi/m_xi);
        }
        sigma_term3.timesEquals(1/m_xi/m_xi);
        sigma_term1.plusEquals(sigma_term2.plusEquals(sigma_term3));

        Matrix invsMtx = sigma_term1.inverse();
        user.m_sigma_u = invsMtx.getArray();

        Utils.add2Array(mu_term1, mu_term2, 1);
        Utils.add2Array(mu_term1, mu_term3, 1);
        Matrix mu_matrix = new Matrix(mu_term1, 1);
        user.m_mu_u = invsMtx.times(mu_matrix).getArray()[0];
    }

    // update mean for pair-wise affinity \mu^{delta_{ij}}, \sigma^{\delta_{ij}} -- Eq(75)
    protected void update_delta_ij_mu(_User4EUB user){
        int i = m_usersIndex.get(user.getUserID());
        int[] iflag = {0}, iprint = {-1, 3};
        double fValue = 0, oldFValue = Double.MAX_VALUE;

        double[] muG = new double[m_users.size()];
        double[] diag = new double[m_users.size()];
        HashSet<Integer> friends = m_userI2JMap.get(i);

        try {
            do {
                Arrays.fill(muG, 0);
                Arrays.fill(diag, 0);
                for(int j=0; j<m_users.size(); j++){
                    if (i == j) continue;
                    int eij = friends.contains(j) ? 1 : 0;
                    double[] fgValue = calcFGValueDeltaMu(user, eij, j);
                    fValue += fgValue[0];
                    muG[j] = fgValue[1];
                }
                // gradient test
                System.out.println("FValue is: " + fValue);
                if (fValue<oldFValue)
                    System.out.print("o");
                else
                    System.out.print("x");
                LBFGS.lbfgs(muG.length, 6, user.m_mu_delta, fValue, muG, false, diag, iprint, 1e-3, 1e-16, iflag);
                oldFValue = fValue;

            } while(iflag[0] != 0);
            System.out.println();
        } catch(Exception e){
            System.err.println("LBFGS fails!!!!");
            e.printStackTrace();
        }
    }

    protected double[] calcFGValueDeltaMu(_User4EUB user, int eij, int j){
        double[] mu = user.m_mu_delta, sigma = user.m_sigma_delta;
        double dotProd = Utils.dotProduct(user.m_mu_u, m_users.get(j).m_mu_u);

        double fValue = eij == 1 ? mu[j] : 0;
        double gValue = eij;
        double term1 = -1/user.m_epsilon[j] * Math.exp(mu[j] + 0.5 * sigma[j] * sigma[j]);

        fValue += term1 - 0.5/user.m_epsilon[j]/user.m_epsilon[j] * (mu[j] * mu[j] + sigma[j] * sigma[j] - 2 * mu[j] * dotProd);
        gValue += term1 - 1/user.m_epsilon[j]/user.m_epsilon[j] * (mu[j] - dotProd);
        return new double[]{fValue, gValue};
    }

    // update variance for pair-wise affinity \mu^{delta_{ij}}, \sigma^{\delta_{ij}} -- Eq(75)
    protected void update_delta_ij_sigma(_User4EUB user){
        int i = m_usersIndex.get(user.getUserID());
        int[] iflag = {0}, iprint = {-1, 3};
        double fValue = 0, oldFValue = Double.MAX_VALUE;

        double[] sigmaG = new double[m_users.size()];
        double[] diag = new double[m_users.size()];

        try {
            do {
                Arrays.fill(sigmaG, 0);
                Arrays.fill(diag, 0);
                for(int j=0; j<m_users.size(); j++){
                    if (i == j) continue;
                    double[] fgValue = calcFGValueDeltaSigma(user, j);
                    fValue += fgValue[0];
                    sigmaG[j] = fgValue[1];
                }
                // gradient test
                System.out.println("FValue is: " + fValue);
                if (fValue<oldFValue)
                    System.out.print("o");
                else
                    System.out.print("x");
                LBFGS.lbfgs(sigmaG.length, 6, user.m_mu_delta, fValue, sigmaG, false, diag, iprint, 1e-3, 1e-16, iflag);
                oldFValue = fValue;

            } while(iflag[0] != 0);
            System.out.println();
        } catch(Exception e){
            System.err.println("LBFGS fails!!!!");
            e.printStackTrace();
        }
    }

    protected double[] calcFGValueDeltaSigma(_User4EUB user, int j){

        double[] mu = user.m_mu_delta, sigma = user.m_sigma_delta;
        double term1 = Math.exp(mu[j] + 0.5 * sigma[j] * sigma[j]);
        double fValue = -1/user.m_epsilon[j] * term1 - 0.5/user.m_epsilon[j]/user.m_epsilon[j] *
                (mu[j] * mu[j] + sigma[j] * sigma[j]) + Math.log(sigma[j]);
        double gValue = -sigma[j]/user.m_epsilon[j] * term1 - sigma[j]/user.m_epsilon[j]/user.m_epsilon[j] + 1/sigma[j];
        return new double[]{fValue, gValue};
    }

    // update the taylor parameter epsilon
    protected void update_epsilon(_User4EUB user){
        int i = m_usersIndex.get(user.getUserID());
        for(int j=0; j<user.m_epsilon.length; j++){
            if (j != i)
                user.m_epsilon[j] = Math.exp(user.m_mu_delta[j] + 0.5 * user.m_sigma_delta[j] * user.m_sigma_delta[j]) + 1;
        }
    }

    protected void update_eta_id(_Doc4EUB doc){
        double logSum;
        _SparseFeature[] fvs = doc.getSparse();
        Arrays.fill(doc.m_sstat, 0);

        for(int n=0; n<fvs.length; n++){
            int v = fvs[n].getIndex();
            for(int k=0; k<number_of_topics; k++){
                // Eq(86) update eta of each document
                doc.m_phi[n][k] = doc.m_mu_theta[k] + topic_term_probabilty[k][v];
            }
            // normalize
            logSum = Utils.logSum(doc.m_phi[n]);
            for(int k=0; k<number_of_topics; k++){
                doc.m_phi[n][k] = Math.exp(doc.m_phi[n][k] - logSum);
                // update \sum_\eta_{vk}, only related with topic index k
                doc.m_sstat[k] += fvs[n].getValue() * doc.m_phi[n][k];
            }
        }
    }

    // variational parameters for topic distribution \theta_{id}
    protected void update_theta_id_mu(_Doc4EUB doc){
        // user index of the current doc
        int i = m_docUserMap.get(doc.getID());
        int N = doc.getTotalDocLength();
        int[] iflag = {0}, iprint = {-1, 3};
        double fValue = 0, oldFValue = Double.MAX_VALUE, dotProd, moment;
        double[] muG = new double[number_of_topics];
        double[] diag = new double[number_of_topics];

        try{
            do{
                Arrays.fill(muG, 0);
                Arrays.fill(diag, 0);
                for(int k=0; k<number_of_topics; k++){
                    // function value
                    moment = N * Math.exp(doc.m_mu_theta[k] + 0.5 * doc.m_sigma_theta[k] - doc.m_logZeta);
                    fValue += -0.5 * m_tau * doc.m_mu_theta[k] * doc.m_mu_theta[k];
                    dotProd = Utils.dotProduct(m_topics.get(k).m_mu_phi, m_users.get(i).m_mu_u);
                    fValue += m_tau * doc.m_mu_theta[k] * dotProd + doc.m_mu_theta[k] * doc.m_sstat[k] - moment;
                    // gradient
                    muG[k] = -m_tau * doc.m_mu_theta[k] + m_tau * dotProd + doc.m_sstat[k] - moment;
                }
                // gradient test
                System.out.println("FValue is: " + fValue);
                if (fValue<oldFValue)
                    System.out.print("o");
                else
                    System.out.print("x");
                LBFGS.lbfgs(muG.length, 6, doc.m_mu_theta, fValue, muG, false, diag, iprint, 1e-3, 1e-16, iflag);
                oldFValue = fValue;

            } while(iflag[0] != 0);
            System.out.println();
        } catch(Exception e){
            System.err.println("LBFGS fails!!!!");
            e.printStackTrace();
        }
    }

    protected void update_theta_id_sigma(_Doc4EUB doc){
        int N = doc.getTotalDocLength();
        int[] iflag = {0}, iprint = {-1, 3};
        double fValue = 0, oldFValue = Double.MAX_VALUE, dotProd, moment;
        double[] sigmaG = new double[number_of_topics];
        double[] diag = new double[number_of_topics];

        try{
            do{
                Arrays.fill(sigmaG, 0);
                Arrays.fill(diag, 0);
                for(int k=0; k<number_of_topics; k++){
                    // function value
                    moment = N * Math.exp(doc.m_mu_theta[k] + 0.5 * doc.m_sigma_theta[k] - doc.m_logZeta);
                    fValue += -0.5 * m_tau * doc.m_sigma_theta[k] - moment + 0.5 * Math.log(doc.m_sigma_theta[k]);
                    // gradient
                    sigmaG[k] = -0.5 * m_tau - 0.5 * moment + 0.5/doc.m_sigma_theta[k];
                }
                // gradient test
                System.out.println("FValue is: " + fValue);
                if (fValue<oldFValue)
                    System.out.print("o");
                else
                    System.out.print("x");
                LBFGS.lbfgs(sigmaG.length, 6, doc.m_sigma_theta, fValue, sigmaG, false, diag, iprint, 1e-3, 1e-16, iflag);
                oldFValue = fValue;

            } while(iflag[0] != 0);
            System.out.println();
        } catch(Exception e){
            System.err.println("LBFGS fails!!!!");
            e.printStackTrace();
        }
    }

    // taylor parameter
    protected void update_zeta(_Doc4EUB doc){
        doc.m_logZeta = doc.m_mu_theta[0] + 0.5 * doc.m_sigma_theta[0];

        for(int k=1; k<number_of_topics; k++){
            doc.m_logZeta = Utils.logSum(doc.m_logZeta, doc.m_mu_theta[k] + 0.5 * doc.m_sigma_theta[k]);
        }
    }

    protected double calc_log_likelihood_per_topic(_Topic4EUB topic){
        return 0;
    }

    protected double calc_log_likelihood_per_doc(_Doc4EUB doc){
        return 0;
    }

    protected double calc_log_likelihood_per_user(_User4EUB user){
        return 0;
    }

    // cross validation

    // one fold validation

    // currently, assume we have train/test data

    public static void main(String[] args){
        double[] a = new double[]{1, 2, 3};
        Matrix aMtx = new Matrix(a, 1);
        double[][] b = aMtx.getArray();
        for(int i=0; i<b.length; i++){
            System.out.println(i);
            for(int j=0; j<b[i].length; j++){
                System.out.println(b[i][j]);
            }
        }
    }
}