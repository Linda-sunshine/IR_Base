package topicmodels.UserEmbedding;

import Jama.Matrix;
import LBFGS.LBFGS;
import org.apache.commons.math3.distribution.BinomialDistribution;
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

    protected int m_embeddingDim;

    protected HashMap<String, Integer> m_usersIndex;
    // key: user index, value: document index array
    protected HashMap<Integer, ArrayList<Integer>> m_userDocMap;
    // assume i follows ps, key: index i, value: index ps
    protected HashMap<Integer, HashSet<Integer>> m_userI2PMap;
    // assume i does not follow p', key: index i, value: index p'
    protected HashMap<Integer, HashSet<Integer>> m_userI2PPrimeMap;

    // assume qs follows i, key: index i, value: index qs
    protected HashMap<Integer, HashSet<Integer>> m_userQ2IMap;
    // assume q' does not follow i, key: index q', value: index i
    protected HashMap<Integer, HashSet<Integer>> m_userQPrime2IMap;

    // key: doc index, value: user index
    protected HashMap<Integer, Integer> m_docUserMap;

    protected BinomialDistribution m_bernoulli;

    /*****variational parameters*****/
    protected double t_mu = 0.1, t_sigma = 0.1;
    protected double d_mu = 0.1, d_sigma = 0.1;
    protected double u_mu = 0.1, u_sigma = 0.1;

    /*****model parameters*****/
    // this alpha is differnet from alpha in LDA
    // alpha is precision parameter for topic embedding in EUB
    // alpha is a vector parameter for dirichlet distribution
    protected double m_alpha_s = 1.0;
    protected double m_tau = 1;
    protected double m_gamma = 1.0;
    protected double m_xi = 2;

    /*****Sparsity parameter******/
    protected double m_rho = 0.01;

    // Decide if the network is directional or not
    private boolean m_directionFlag = false;

    protected int m_displayLv = 2;

    // For debugging purpose, record how many failure in variational inference.
    protected int m_muFailCounter = 0, m_sigmaFailCounter = 0;

    public EUB(int number_of_iteration, double converge, double beta,
               _Corpus c, double lambda, int number_of_topics, double alpha,
               int varMaxIter, double varConverge, int m) {
        super(number_of_iteration, converge, beta, c, lambda, number_of_topics, alpha,
                varMaxIter, varConverge);
        m_embeddingDim = m;
        m_topics = new ArrayList<>();
        m_users = new ArrayList<>();
        m_docs = new ArrayList<>();

        m_bernoulli = new BinomialDistribution(1, m_rho);

    }

    // Load the data for later user
    public void buildLookupTables(ArrayList<_User> users){
        m_usersIndex = new HashMap<>();
        m_userDocMap = new HashMap<>();
        m_docUserMap = new HashMap<>();

        m_userI2PMap = new HashMap<>();
        m_userI2PPrimeMap = new HashMap<>();
        m_userQ2IMap = new HashMap<>();
        m_userQPrime2IMap = new HashMap<>();

        // build the lookup table for user/doc/topic
        for(int i=0; i< users.size(); i++){
            _User4EUB user = new _User4EUB(users.get(i));
            m_users.add(user);
            m_usersIndex.put(user.getUserID(), i);
            buildUserDocs(i, user, users.get(i).getReviews());
        }

        for(int k=0; k<number_of_topics; k++){
            m_topics.add(new _Topic4EUB(k));
        }
    }

    protected void buildUserDocs(int i, _User4EUB user, ArrayList<_Review> reviews){
        m_userDocMap.put(i, new ArrayList<>());
        ArrayList<_Doc4EUB> docs = new ArrayList<>();
        for(_Review r: reviews){
            int dIndex = m_docs.size();
            _Doc4EUB doc = new _Doc4EUB(r, dIndex);
            docs.add(doc);
            m_docs.add(doc);
            m_userDocMap.get(i).add(dIndex);
            m_docUserMap.put(dIndex, i);
        }
        user.setReviews(docs);
    }

    @Override
    public void EMonCorpus(){

        m_trainSet = new ArrayList<>();
        // collect all training reviews
        for(_User u: m_users){
            for(_Review r: u.getReviews()){
                m_trainSet.add(r);
            }
        }
        EM();
    }

    int m_EStepCount = 0;
    @Override
    public void EM() {

        // sample non-interactions first before E-step
        constructNetwork();
        initialize_probability(m_trainSet);

        int iter = 0;
        double lastAllLikelihood = 1.0;
        double currentAllLikelihood;
        double converge;
        do {
            if(++m_EStepCount % 5 == 0)
                constructNetwork();

            currentAllLikelihood = E_step();
            System.out.format("[Info]The current loglikelihood is: %.5f.\n", currentAllLikelihood);
            if (iter >= 0)
                converge = Math.abs((lastAllLikelihood - currentAllLikelihood) / lastAllLikelihood);
            else
                converge = 1.0;

            calculate_M_step(iter);

            lastAllLikelihood = currentAllLikelihood;
        } while( ++iter < number_of_iteration && converge > m_converge);
    }

    // put the user interaction and non-interaction information in the four hashmaps
    protected void constructNetwork() {
        System.out.print("[Info] Start sampling non-interactions....");
        m_userI2PMap.clear();
        m_userQ2IMap.clear();
        m_userI2PPrimeMap.clear();
        m_userQPrime2IMap.clear();

        if (!m_directionFlag){
            for(String uiId: m_usersIndex.keySet()){
                int uiIndex = m_usersIndex.get(uiId);
                // interactions
                sampleNonDirectionalInteractions(uiIndex);
                // non-interactions
                sampleNonDirectionalNonInteractions(uiIndex);
            }
        } else{
            for(String uiId: m_usersIndex.keySet()){
                int uiIndex = m_usersIndex.get(uiId);
                // interactions
                sampleDirectionalInteractions(uiIndex);
                // non-interactions
                sampleDirectionalNonInteractions(uiIndex);
            }
        }
        System.out.println("Done");
    }

    protected void sampleDirectionalInteractions(int uiIndex){
        // interactions
        if(!m_userI2PMap.containsKey(uiIndex))
            m_userI2PMap.put(uiIndex, new HashSet<>());

        _User4EUB ui = m_users.get(uiIndex);
        if(ui.getFriends() != null && ui.getFriends().length > 0){
            for(String ujId: ui.getFriends()) {
                int ujIndex = m_usersIndex.get(ujId);
                if(!m_userQ2IMap.containsKey(ujIndex))
                    m_userQ2IMap.put(ujIndex, new HashSet<>());

                // eij
                m_userI2PMap.get(uiIndex).add(ujIndex);
                m_userQ2IMap.get(ujIndex).add(uiIndex);
            }
        }
    }

    protected void sampleDirectionalNonInteractions(int uiIndex){

        // sample non-interactions
        HashSet<Integer> nonInteactions = new HashSet<>(m_usersIndex.values());
        nonInteactions.remove(uiIndex);

        if(m_userI2PMap.containsKey(uiIndex)){
            for(int p: m_userI2PMap.get(uiIndex))
                nonInteactions.remove(p);
        }

        if (!m_userI2PPrimeMap.containsKey(uiIndex))
            m_userI2PPrimeMap.put(uiIndex, new HashSet<>());

        for(int ujIndex: nonInteactions){
            if (m_bernoulli.sample() == 1) {
                if (!m_userQPrime2IMap.containsKey(ujIndex))
                    m_userQPrime2IMap.put(ujIndex, new HashSet<>());
                //eij=0
                m_userI2PPrimeMap.get(uiIndex).add(ujIndex);
                m_userQPrime2IMap.get(ujIndex).add(uiIndex);
            }
        }
    }

    protected void sampleNonDirectionalInteractions(int uiIndex){

        _User4EUB ui = m_users.get(uiIndex);
        if(ui.getFriends() != null && ui.getFriends().length > 0){
            // interactions
            if(!m_userI2PMap.containsKey(uiIndex))
                m_userI2PMap.put(uiIndex, new HashSet<>());
            if(!m_userQ2IMap.containsKey(uiIndex))
                m_userQ2IMap.put(uiIndex, new HashSet<>());

            for(String ujId: ui.getFriends()) {
                int ujIndex = m_usersIndex.get(ujId);
                if(!m_userI2PMap.containsKey(ujIndex))
                    m_userI2PMap.put(ujIndex, new HashSet<>());
                if(!m_userQ2IMap.containsKey(ujIndex))
                    m_userQ2IMap.put(ujIndex, new HashSet<>());

                // eij
                m_userI2PMap.get(uiIndex).add(ujIndex);
                m_userQ2IMap.get(ujIndex).add(uiIndex);

                // eji
                m_userI2PMap.get(ujIndex).add(uiIndex);
                m_userQ2IMap.get(uiIndex).add(ujIndex);
            }
        }
    }

    protected void sampleNonDirectionalNonInteractions(int uiIndex) {
        // sample non-interactions
        HashSet<Integer> nonInteactions = new HashSet<>(m_usersIndex.values());
        nonInteactions.remove(uiIndex);

        if(m_userI2PMap.containsKey(uiIndex)){
           for(int p: m_userI2PMap.get(uiIndex))
               nonInteactions.remove(p);
        }

        if (!m_userI2PPrimeMap.containsKey(uiIndex))
            m_userI2PPrimeMap.put(uiIndex, new HashSet<>());
        if (!m_userQPrime2IMap.containsKey(uiIndex))
            m_userQPrime2IMap.put(uiIndex, new HashSet<>());

        for(int ujIndex: nonInteactions){
            if (m_bernoulli.sample() == 1) {
                if (!m_userI2PPrimeMap.containsKey(ujIndex))
                    m_userI2PPrimeMap.put(ujIndex, new HashSet<>());
                if (!m_userQPrime2IMap.containsKey(ujIndex))
                    m_userQPrime2IMap.put(ujIndex, new HashSet<>());
                //eij=0
                m_userI2PPrimeMap.get(uiIndex).add(ujIndex);
                m_userQPrime2IMap.get(ujIndex).add(uiIndex);

                //eji=0
                m_userI2PPrimeMap.get(ujIndex).add(uiIndex);
                m_userQPrime2IMap.get(uiIndex).add(ujIndex);
            }
        }
    }

    @Override
    protected void initialize_probability(Collection<_Doc> docs) {

        System.out.println("[Info]Initializing topics, documents and users...");
        for(_Topic4EUB t: m_topics)
             t.setTopics4Variational(m_embeddingDim, t_mu, t_sigma);

        for(_Doc d: m_trainSet)
            ((_Doc4EUB) d).setTopics4Variational(number_of_topics, d_alpha, d_mu, d_sigma);

        for(_User4EUB u: m_users)
            u.setTopics4Variational(m_embeddingDim, m_users.size(), u_mu, u_sigma, m_xi);

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

            m_muFailCounter = 0;
            m_sigmaFailCounter = 0;
            int count = 0;
            for (_Doc doc: m_trainSet) {
                count++;
                if(count % 100 == 0)
                    System.out.print("*");
                if(count % 10000 == 0)
                    System.out.println();
                totalLikelihood += varInference4Doc((_Doc4EUB) doc);
                if(Double.isNaN(totalLikelihood) || Double.isInfinite(totalLikelihood))
                    System.out.println("[error] The likelihood is Nan or Infinity!!");
            }
            System.out.format("\n[Stat]In VI for doc, lbfgs fails %d/%d times for mu, %d times for sigma!\n", m_muFailCounter, m_trainSet.size(), m_sigmaFailCounter);
            for (_Topic4EUB topic: m_topics) {
                totalLikelihood += varInference4Topic(topic);
                if(Double.isNaN(totalLikelihood) || Double.isInfinite(totalLikelihood))
                    System.out.println("[error] The likelihood is Nan or Infinity!!");
            }

            m_muFailCounter = 0;
            m_sigmaFailCounter = 0;
            count = 0;
            for(_User4EUB user: m_users){
                count++;
                if(count % 100 == 0)
                    System.out.print("*");
                if(count % 1000 == 0)
                    System.out.println();
                totalLikelihood += varInference4User(user);
                if(Double.isNaN(totalLikelihood) || Double.isInfinite(totalLikelihood))
                    System.out.println("[error] The likelihood is Nan or Infinity!!");
            }
            System.out.format("\n[Stat]In VI for users, lbfgs fails %d/%d times for mu, %d times for sigma!\n", m_muFailCounter, m_users.size(), m_sigmaFailCounter);

            if(Double.isNaN(totalLikelihood) || Double.isInfinite(totalLikelihood))
                System.out.println("[error] The likelihood is Nan or Infinity!!");

            if(iter > 0)
                converge = Math.abs((totalLikelihood - last) / last);
            else
                converge = 1.0;

            last = totalLikelihood;

            System.out.format("[Info]Single-thread E-Step: %d iteration, likelihood=%.2f, converge to %.8f\n",
                    iter, last, converge);

        }while(iter++ < m_varMaxIter && converge > m_varConverge);

        for(_Doc doc: m_trainSet)
            updateStats4Doc((_Doc4EUB) doc);

        return totalLikelihood;
    }

    @Override
    public void calculate_M_step(int iter){

//        est_alpha();// precision for topic embedding
//        est_gamma(); // precision for user embedding
        est_beta(); // topic-word distribution
//        est_tau(); // precision for topic proportion
//        est_xi(); // sigma for the affinity \delta_{ij}
    }

    protected void est_alpha(){
        double denominator = 0;
        for(int k=0; k<number_of_topics; k++){
            _Topic4EUB topic = m_topics.get(k);
            denominator += sumSigmaDiagAddMuTransposeMu(topic.m_sigma_phi, topic.m_mu_phi);
        }
        m_alpha_s = denominator!=0 ? (number_of_topics * m_embeddingDim / denominator) : 0;
    }

    protected void est_gamma(){
        double denominator = 0;
        for(int uIndex: m_userDocMap.keySet()){
            _User4EUB user = m_users.get(uIndex);
            denominator += sumSigmaDiagAddMuTransposeMu(user.m_sigma_u, user.m_mu_u);
        }
        m_gamma = denominator!=0 ? (m_users.size() * m_embeddingDim) / denominator : 0;
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
        double D = m_trainSet.size(); // total number of documents
        for(_Doc d: m_trainSet){
            _Doc4EUB doc = (_Doc4EUB) d;
            int uIndex = m_docUserMap.get(doc.getIndex());
            int uIdx = m_usersIndex.get(doc.getUserID());
            if(uIndex != uIdx)
                System.out.println("Error!!!");
            _User4EUB user = m_users.get(uIndex);
            denominator += calculateStat4Tau(user, doc);
        }
        m_tau = denominator != 0 ? D * number_of_topics / denominator : 0;
    }

    protected void est_xi(){
        double xiSquare = 0;
        // sum over all interactions and non-interactions
        // i->p and p'
        for(int i=0; i<m_users.size(); i++){
            _User4EUB ui = m_users.get(i);

            if(m_userI2PMap.containsKey(i)){
                HashSet<Integer> ps = m_userI2PMap.get(i);
                for(int p: ps)
                    xiSquare += calculateStat4Xi(ui, m_users.get(p), p);
            }
            if(m_userI2PPrimeMap.containsKey(i)){
                HashSet<Integer> pPrimes = m_userI2PPrimeMap.get(i);
                for(int pPrime: pPrimes)
                    xiSquare += calculateStat4Xi(ui, m_users.get(pPrime), pPrime);
            }
            if(m_userQ2IMap.containsKey(i)){
                HashSet<Integer> qs = m_userQ2IMap.get(i);
                for(int q: qs)
                    xiSquare += calculateStat4Xi(m_users.get(q), ui, i);
            }
            if(m_userQPrime2IMap.containsKey(i)){
                HashSet<Integer> qPrimes = m_userQPrime2IMap.get(i);
                for(int qPrime: qPrimes)
                    xiSquare += calculateStat4Xi(m_users.get(qPrime), ui, i);
            }
        }
        m_xi = (xiSquare > 0) ? Math.sqrt(xiSquare) : 0;

    }

    protected double calculateStat4Xi(_User4EUB ui, _User4EUB uj, int j){
        double val = ui.m_mu_delta[j] * ui.m_mu_delta[j] + ui.m_sigma_delta[j] * ui.m_sigma_delta[j];
        for(int m=0; m<m_embeddingDim; m++){
            val += -2 * ui.m_mu_delta[j] * ui.m_mu_u[m] * uj.m_mu_u[m];
            for(int l=0; l<m_embeddingDim; l++){
                val += (ui.m_sigma_u[m][l] + ui.m_mu_u[m] * uj.m_mu_u[l]) *
                        (uj.m_sigma_u[m][l] + uj.m_mu_u[m] * uj.m_mu_u[l]);
            }
        }
        return val;
    }

    protected double calculateStat4Tau(_User4EUB user, _Doc4EUB doc){
        double term1 = 0, term2 = 0, term3 = 0;

        for(int k=0; k<number_of_topics; k++){
            _Topic4EUB topic = m_topics.get(k);
            term1 += doc.m_sigma_theta[k] + doc.m_mu_theta[k] * doc.m_mu_theta[k];
            for(int m=0 ; m<m_embeddingDim; m++){
                term2 += 2 * doc.m_mu_theta[k] * topic.m_mu_phi[m] * user.m_mu_u[m];
                for(int l=0; l<m_embeddingDim; l++){
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
        // update the variational parameters for user embedding ui -- Eq(70) and Eq(72)
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

        Matrix term1 = new Matrix(new double[m_embeddingDim][m_embeddingDim]);
        double[] term2 = new double[m_embeddingDim];

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
        double[][] diag = new double[m_embeddingDim][m_embeddingDim];
        for(int i=0; i<m_embeddingDim; i++){
            diag[i][i] = m_alpha_s;
        }
        // + \alpha * I
        term1.plusEquals(new Matrix(diag));
        Matrix invsMtx = term1.inverse();

        topic.m_sigma_phi = invsMtx.getArray();

        topic.m_mu_phi = matrixMultVector(topic.m_sigma_phi, term2);
    }

    protected double[] matrixMultVector(double[][] mtx, double[] vct){
        if(mtx[0].length != vct.length)
            return null;
        double[] res = new double[mtx.length];
        for(int i=0; i<mtx.length; i++){
            res[i] = Utils.dotProduct(mtx[i], vct);
        }
        return res;
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

    // \sum_{mm}\simga[m][m] + \mu^T * \mu
    protected double sumSigmaDiagAddMuTransposeMu(double[] sigma, double[] mu){
        if(sigma.length != mu.length)
            return 0;
        int dim = mu.length;
        double sum = 0;
        for(int m=0; m<dim; m++){
            sum += sigma[m] + mu[m] * mu[m];
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

    // update the variational parameters for user embedding ui -- Eq(44) and Eq(45)
    protected void update_u_i(_User4EUB ui){
        // termT: the part related with user's topics
        Matrix sigma_termT = new Matrix(new double[m_embeddingDim][m_embeddingDim]);
        double[] mu_termT = new double[m_embeddingDim];
        double[] mu_termU = new double[m_embeddingDim];

        int i = m_usersIndex.get(ui.getUserID());
        int docSize = m_userDocMap.get(i).size();
        for(int k=0; k<number_of_topics; k++){
            // stat for updating sigma
            _Topic4EUB topic = m_topics.get(k);
            Matrix tmp = new Matrix(sigmaAddMuMuTranspose(topic.m_sigma_phi, topic.m_mu_phi));
            sigma_termT.plusEquals(tmp.timesEquals(docSize));

            // stat for updating mu
            for(int dIndex: m_userDocMap.get(i)){
                _Doc4EUB doc = m_docs.get(dIndex);
                Utils.add2Array(mu_termT, topic.m_mu_phi, m_tau * doc.m_mu_theta[topic.getIndex()]);
            }
        }
        // * \tau
        sigma_termT.timesEquals(m_tau);

        // \gamma * I
        double[][] diag = new double[m_embeddingDim][m_embeddingDim];
        for(int a=0; a<m_embeddingDim; a++){
            diag[a][a] = m_gamma;
        }
        // + \gamma * I
        sigma_termT.plusEquals(new Matrix(diag));


        // termP: user_i -> user_p
        if(m_userI2PMap.containsKey(i)){
            updateI2JStat(m_userI2PMap, ui, i, mu_termU, sigma_termT);
        }

        // termPPrime: user_i -> user_p'
        if(m_userI2PPrimeMap.containsKey(i)){
            updateI2JStat(m_userI2PPrimeMap, ui, i, mu_termU, sigma_termT);
        }

        // termQ: user_q -> user_i
        if(m_userQ2IMap.containsKey(i)){
            updateJ2IStat(m_userQ2IMap, ui, i, mu_termU, sigma_termT);
        }

        // termQPrime: user_q' -> user_i
        if(m_userQPrime2IMap.containsKey(i)){
            updateJ2IStat(m_userQPrime2IMap, ui, i, mu_termU, sigma_termT);
        }

        Matrix invsMtx = sigma_termT.inverse();
        ui.m_sigma_u = invsMtx.getArray();

        Utils.add2Array(mu_termT, mu_termU, 1);
        ui.m_mu_u = matrixMultVector(ui.m_sigma_u, mu_termT);
    }

    // i->p or i->p'
    protected void updateI2JStat(HashMap<Integer, HashSet<Integer>> map, _User4EUB ui, int i, double[] mu_termU, Matrix sigma_termT){
        Matrix sigma_termJ = new Matrix(new double[m_embeddingDim][m_embeddingDim]);
        for(int j: map.get(i)){
            _User4EUB uj = m_users.get(j);
            double delta_ij = ui.m_mu_delta[j];
            sigma_termJ.plusEquals(new Matrix(sigmaAddMuMuTranspose(uj.m_sigma_u, uj.m_mu_u)));
            Utils.add2Array(mu_termU, uj.m_mu_u, delta_ij/m_xi/m_xi);
        }
        sigma_termJ.timesEquals(1/m_xi /m_xi);
        sigma_termT.plusEquals(sigma_termJ);
    }

    // q->i or q'->i
    protected void updateJ2IStat(HashMap<Integer, HashSet<Integer>> map, _User4EUB ui, int i, double[] mu_termU, Matrix sigma_termT){
        Matrix sigma_termJ = new Matrix(new double[m_embeddingDim][m_embeddingDim]);
        for(int j: map.get(i)){
            _User4EUB uj = m_users.get(j);
            double delta_ji = uj.m_mu_delta[i];
            sigma_termJ.plusEquals(new Matrix(sigmaAddMuMuTranspose(ui.m_sigma_u, ui.m_mu_u)));
            Utils.add2Array(mu_termU, ui.m_mu_u, delta_ji/m_xi/m_xi);
        }
        sigma_termJ.timesEquals(1/m_xi/m_xi);
        sigma_termT.plusEquals(sigma_termJ);
    }

    // update mean for pair-wise affinity \mu^{delta_{ij}}, \sigma^{\delta_{ij}} -- Eq(47)
    protected void update_delta_ij_mu(_User4EUB ui){
        int i = m_usersIndex.get(ui.getUserID());
        int[] iflag = {0}, iprint = {-1, 3};
        double fValue, oldFValue = Double.MAX_VALUE;

        double[] muG = new double[m_users.size()];
        double[] diag = new double[m_users.size()];
        double[] mu_delta = Arrays.copyOfRange(ui.m_mu_delta, 0, ui.m_mu_delta.length);
        try {
            do {
                Arrays.fill(muG, 0);
                Arrays.fill(diag, 0);
                fValue = 0;
                if(m_userI2PMap.containsKey(i)) {
                    for(int p: m_userI2PMap.get(i)){
                        double[] fgValue = calcFGValueDeltaMu(ui, mu_delta, 1, p);
                        fValue -= fgValue[0];
                        muG[p] -= fgValue[1];
                    }
                }

                if(m_userI2PPrimeMap.containsKey(i)){
                    for(int pPrime: m_userI2PPrimeMap.get(i)){
                        double[] fgValue = calcFGValueDeltaMu(ui, mu_delta, 0, pPrime);
                        fValue -= fgValue[0];
                        muG[pPrime] -= fgValue[1];
                    }
                }
                printFValue(oldFValue, fValue);
                LBFGS.lbfgs(muG.length, 6, mu_delta, fValue, muG, false, diag, iprint, 1e-3, 1e-16, iflag);
                oldFValue = fValue;
            } while(iflag[0] != 0);
            System.out.println();
            ui.m_mu_delta = mu_delta;
        } catch(Exception e){
            m_muFailCounter++;
            System.err.println("LBFGS fails!!!!");
//            e.printStackTrace();
        }
    }

    protected double[] calcFGValueDeltaMu(_User4EUB ui, double[] mu_delta, int eij, int j){
        double[] sigma_delta = ui.m_sigma_delta;
        double dotProd = Utils.dotProduct(ui.m_mu_u, m_users.get(j).m_mu_u);

        double fValue = eij == 1 ? mu_delta[j] : 0;
        double gValue = eij;
        double term1 = -1/ui.m_epsilon[j] * Math.exp(mu_delta[j] + 0.5 * sigma_delta[j] * sigma_delta[j]);

        fValue += term1 - 0.5/m_xi/m_xi * (mu_delta[j] * mu_delta[j] - 2 * mu_delta[j] * dotProd);
        gValue += term1 - 1/m_xi/m_xi * (mu_delta[j] - dotProd);
        return new double[]{fValue, gValue};
    }

    // update variance for pair-wise affinity \mu^{delta_{ij}}, \sigma^{\delta_{ij}} -- Eq(48)
    protected void update_delta_ij_sigma(_User4EUB ui){
        int i = m_usersIndex.get(ui.getUserID());
        int[] iflag = {0}, iprint = {-1, 3};
        double fValue, oldFValue = Double.MAX_VALUE;

        double[] sigmaG = new double[m_users.size()];
        double[] diag = new double[m_users.size()];
        double[] sigma_delta = Arrays.copyOfRange(ui.m_sigma_delta, 0, ui.m_sigma_delta.length);
        try {
            do {
                Arrays.fill(sigmaG, 0);
                Arrays.fill(diag, 0);
                fValue = 0;
                if(m_userI2PMap.containsKey(i)) {
                    for(int p: m_userI2PMap.get(i)){
                        double[] fgValue = calcFGValueDeltaSigma(ui, sigma_delta, p);
                        fValue -= fgValue[0];
                        sigmaG[p] -= fgValue[1];
                    }
                }

                if(m_userI2PPrimeMap.containsKey(i)){
                    for(int pPrime: m_userI2PPrimeMap.get(i)){
                        double[] fgValue = calcFGValueDeltaSigma(ui, sigma_delta, pPrime);
                        fValue -= fgValue[0];
                        sigmaG[pPrime] -= fgValue[1];
                    }
                }
                printFValue(oldFValue, fValue);
                LBFGS.lbfgs(sigmaG.length, 6, sigma_delta, fValue, sigmaG, false, diag, iprint, 1e-3, 1e-16, iflag);
                oldFValue = fValue;

            } while(iflag[0] != 0);
            System.out.println();
            ui.m_sigma_delta = sigma_delta;
        } catch(Exception e){
            m_sigmaFailCounter++;
            System.err.println("LBFGS fails!!!!");
//            e.printStackTrace();
        }
    }

    protected double[] calcFGValueDeltaSigma(_User4EUB ui, double[] sigma, int j){

        double term1 = -1/ui.m_epsilon[j] * Math.exp(ui.m_mu_delta[j] + 0.5 * sigma[j] * sigma[j]);
        double fValue = term1 - 0.5/m_xi/m_xi * (sigma[j] * sigma[j]) + Math.log(Math.abs(sigma[j]));
        double gValue = sigma[j] * term1 - sigma[j]/m_xi/m_xi + 1/sigma[j];
        return new double[]{fValue, gValue};
    }

    // update the taylor parameter epsilon
    protected void update_epsilon(_User4EUB ui){
        int i = m_usersIndex.get(ui.getUserID());
        for(int j=0; j<ui.m_epsilon.length; j++){
            if (j != i)
                ui.m_epsilon[j] = Math.exp(ui.m_mu_delta[j] + 0.5 * m_xi * m_xi) + 1;
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
        int i = m_docUserMap.get(doc.getIndex());
        int N = doc.getTotalDocLength();
        int[] iflag = {0}, iprint = {-1, 3};
        double fValue, oldFValue = Double.MAX_VALUE, dotProd, moment;
        double[] muG = new double[number_of_topics];
        double[] diag = new double[number_of_topics];
        double[] mu_theta = Arrays.copyOfRange(doc.m_mu_theta, 0, doc.m_mu_theta.length);

        try{
            do{
                Arrays.fill(muG, 0);
                Arrays.fill(diag, 0);
                fValue = 0;
                for(int k=0; k<number_of_topics; k++){
                    // function value
                    moment = N * Math.exp(mu_theta[k] + 0.5 * doc.m_sigma_theta[k] - doc.m_logZeta);
                    fValue -= -0.5 * m_tau * mu_theta[k] * mu_theta[k];
                    dotProd = Utils.dotProduct(m_topics.get(k).m_mu_phi, m_users.get(i).m_mu_u);
                    fValue -= m_tau * mu_theta[k] * dotProd + mu_theta[k] * doc.m_sstat[k] - moment;
                    // gradient
                    muG[k] = -(-m_tau * mu_theta[k] + m_tau * dotProd + doc.m_sstat[k] - moment);
                }
                printFValue(oldFValue, fValue);
                LBFGS.lbfgs(muG.length, 6, mu_theta, fValue, muG, false, diag, iprint, 1e-3, 1e-16, iflag);
                oldFValue = fValue;
            } while(iflag[0] != 0);
            System.out.println();
            doc.m_mu_theta = mu_theta;
        } catch(Exception e){
            m_muFailCounter++;
            System.err.println("LBFGS fails!!!!");
            e.printStackTrace();
        }
    }

    protected void printFValue(double oldFValue, double fValue){

        if (m_displayLv==2) {
            System.out.println("Fvalue is " + fValue);
        } else if (m_displayLv==1) {
            if (fValue<oldFValue)
                System.out.print("o");
            else
                System.out.print("x");
        } else if(m_displayLv==0)
            return;
    }

    protected void update_theta_id_sigma(_Doc4EUB doc){
        int N = doc.getTotalDocLength();
        int[] iflag = {0}, iprint = {-1, 3};
        double fValue = 0, oldFValue = Double.MAX_VALUE, dotProd, moment;
        double[] sigmaSqrtG = new double[number_of_topics];
        double[] diag = new double[number_of_topics];
        double[] sigma_sqrt_theta = Arrays.copyOfRange(doc.m_sigma_theta, 0, doc.m_sigma_theta.length);
        try{
            do{
                Arrays.fill(sigmaSqrtG, 0);
                Arrays.fill(diag, 0);
                fValue = 0;
                for(int k=0; k<number_of_topics; k++){
                    // function value
                    moment = N * Math.exp(doc.m_mu_theta[k] + 0.5 * sigma_sqrt_theta[k]
                            * sigma_sqrt_theta[k] - doc.m_logZeta);
                    fValue -= -0.5 * m_tau * sigma_sqrt_theta[k] * sigma_sqrt_theta[k] - moment
                            + 0.5 * Math.log(sigma_sqrt_theta[k] * sigma_sqrt_theta[k]);
                    // gradient
                    sigmaSqrtG[k] = m_tau * sigma_sqrt_theta[k] + sigma_sqrt_theta[k] * moment
                            - 1/sigma_sqrt_theta[k];

                }
                printFValue(oldFValue, fValue);
                LBFGS.lbfgs(sigmaSqrtG.length, 6, sigma_sqrt_theta, fValue, sigmaSqrtG, false, diag, iprint, 1e-3, 1e-16, iflag);
                oldFValue = fValue;
            } while(iflag[0] != 0);
            System.out.println();
            doc.m_sigma_sqrt_theta = sigma_sqrt_theta;
            for(int k=0; k<number_of_topics; k++)
                doc.m_sigma_theta[k] = doc.m_sigma_sqrt_theta[k] * doc.m_sigma_sqrt_theta[k];
        } catch(Exception e){
            m_sigmaFailCounter++;
            System.err.println("LBFGS fails!!!!");
//            e.printStackTrace();
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

        double logLikelihood = 0.5 * m_embeddingDim * (Math.log(m_alpha_s) + 1);
        double determinant = new Matrix(topic.m_sigma_phi).det();
        logLikelihood += 0.5 * Math.log(Math.abs(determinant)) - 0.5 * m_alpha_s *
                sumSigmaDiagAddMuTransposeMu(topic.m_sigma_phi, topic.m_mu_phi);
        return logLikelihood;
    }

    protected double calc_log_likelihood_per_doc(_Doc4EUB doc){
        // the first part
        double logLikelihood = 0.5 * number_of_topics * (Math.log(m_tau) + 1) - doc.getTotalDocLength() *
                (doc.m_logZeta -1 ) - 0.5 * m_tau * sumSigmaDiagAddMuTransposeMu(doc.m_sigma_theta, doc.m_mu_theta);
        double determinant = 1;
        for(int k=0; k<number_of_topics; k++){
            determinant *= doc.m_sigma_theta[k];
        }
        logLikelihood += 0.5 * Math.log(Math.abs(determinant));
        double term1 = 0, term2 = 0;
        for(int m=0; m<m_embeddingDim; m++){
            for(int k=0; k<number_of_topics; k++){
                _Topic4EUB topic = m_topics.get(k);
                _User4EUB user = m_users.get(m_docUserMap.get(doc.getIndex()));
                term1 += doc.m_mu_theta[k] * topic.m_mu_phi[m] * user.m_mu_u[m];
                for(int l=0; l<m_embeddingDim; l++){
                    term2 += (user.m_sigma_u[m][l] + user.m_mu_u[m] * user.m_mu_u[l])
                            * (topic.m_sigma_phi[m][l] + topic.m_mu_phi[m] * topic.m_mu_phi[l]);
                }
            }
        }
        logLikelihood += m_tau * term1 - 0.5 * m_tau * term2;

        // the second part which involves with words
        _SparseFeature[] fv = doc.getSparse();
        for(int k = 0; k < number_of_topics; k++) {
            for (int n = 0; n < fv.length; n++) {
                int wid = fv[n].getIndex();
                double v = fv[n].getValue() * doc.m_phi[n][k];
                logLikelihood += v * (doc.m_mu_theta[k] - Math.log(doc.m_phi[n][k]) + topic_term_probabilty[k][wid]);
            }
            logLikelihood += -Math.exp(doc.m_mu_theta[k] + 0.5 * doc.m_sigma_theta[k])/Math.exp(doc.m_logZeta);
        }
        return logLikelihood;
    }

    protected double calc_log_likelihood_per_user(_User4EUB user){
        double logLikelihood = 0.5 * m_embeddingDim * (Math.log(m_gamma) + 1) -
                0.5 * m_gamma * sumSigmaDiagAddMuTransposeMu(user.m_sigma_u, user.m_mu_u);

        // calculate the determinant of the matrix
        double determinant = new Matrix(user.m_sigma_u).det();
        logLikelihood += 0.5 * Math.log(Math.abs(determinant));
        if(Double.isNaN(logLikelihood) || Double.isInfinite(logLikelihood))
            System.out.println("[error] The likelihood is Nan or Infinity!!");
        int i = m_usersIndex.get(user.getUserID());
        // case 1: i -> p
        if(m_userI2PMap.containsKey(i)){
            for(int p: m_userI2PMap.get(i)){
                logLikelihood += m_userI2PMap.get(i).size() * (-1/m_xi + 1.5 - 2 * Math.log(m_xi));

                logLikelihood += calcLogLikelihoodOneEdge(user, m_users.get(p), i, p, 1);
            }
        }
        if(Double.isNaN(logLikelihood) || Double.isInfinite(logLikelihood))
            System.out.println("[error] The likelihood is Nan or Infinity!!");

        // case 2: i -> p'
        if(m_userI2PPrimeMap.containsKey(i)){
            for(int pPrime: m_userI2PPrimeMap.get(i)){
                logLikelihood += m_userI2PPrimeMap.get(i).size() * (-1/m_xi + 1.5 - 2 * Math.log(m_xi));
                logLikelihood += calcLogLikelihoodOneEdge(user, m_users.get(pPrime), i, pPrime, 0);
            }
        }
        if(Double.isNaN(logLikelihood) || Double.isInfinite(logLikelihood))
            System.out.println("[error] The likelihood is Nan or Infinity!!");

        // case 3: q -> i
        if(m_userQ2IMap.containsKey(i)){
            for(int q: m_userQ2IMap.get(i)){
                logLikelihood += m_userQ2IMap.get(i).size() * (-1/m_xi + 1.5 - 2 * Math.log(m_xi));
                logLikelihood += calcLogLikelihoodOneEdge(m_users.get(q), user, q, i, 1);
            }
        }
        if(Double.isNaN(logLikelihood) || Double.isInfinite(logLikelihood))
            System.out.println("[error] The likelihood is Nan or Infinity!!");
        // case 4: q' -> i
        if(m_userQPrime2IMap.containsKey(i)) {
            for(int qPrime: m_userQPrime2IMap.get(i)){
                logLikelihood += m_userQPrime2IMap.get(i).size() * (-1/m_xi + 1.5 - 2 * Math.log(m_xi));
                logLikelihood += calcLogLikelihoodOneEdge(m_users.get(qPrime), user, qPrime, i, 0);
            }
        }
        if(Double.isNaN(logLikelihood) || Double.isInfinite(logLikelihood))
            System.out.println("[error] The likelihood is Nan or Infinity!!");
        return logLikelihood;
    }

    protected double calcLogLikelihoodOneEdge(_User4EUB ui, _User4EUB uj, int i, int j, int eij){
        double muDelta = ui.m_mu_delta[j], sigmaDelta = ui.m_sigma_delta[j];
        double logLikelihood = eij * muDelta - 1/m_xi * Math.exp(muDelta + 0.5 * sigmaDelta)
                -(muDelta * muDelta + sigmaDelta * sigmaDelta)/(2 * m_xi * m_xi) + Math.log(Math.abs(sigmaDelta));
        if(Double.isNaN(logLikelihood) || Double.isInfinite(logLikelihood))
            System.out.println("[error] The likelihood is Nan or Infinity!!");
        for(int m=0; m<m_embeddingDim; m++){
            logLikelihood += muDelta/(m_xi * m_xi) * ui.m_mu_u[m] * uj.m_mu_u[m];
            if(Double.isNaN(logLikelihood) || Double.isInfinite(logLikelihood))
                System.out.println("[error] The likelihood is Nan or Infinity!!");
            for(int l=0; l<m_embeddingDim; l++){
                logLikelihood += -1/(2*m_xi*m_xi)*(ui.m_sigma_u[m][l] + ui.m_mu_u[m] * ui.m_mu_u[l]) *
                        (uj.m_sigma_u[m][l] + uj.m_mu_u[m] * uj.m_mu_u[l]);
                if(Double.isNaN(logLikelihood) || Double.isInfinite(logLikelihood))
                    System.out.println("[error] The likelihood is Nan or Infinity!!");
            }
        }
        if(Double.isNaN(logLikelihood) || Double.isInfinite(logLikelihood))
            System.out.println("[error] The likelihood is Nan or Infinity!!");
        return logLikelihood;
    }

    // fixed cross validation
    public void fixedCrossValidation(int kFold){
        m_trainSet = new ArrayList<>();
        m_testSet = new ArrayList<>();
        for(int k=0; k<kFold; k++){
            m_trainSet.clear();
            m_testSet.clear();
            for(int i=0; i<m_users.size(); i++){
                for(_Review r: m_users.get(i).getReviews()){
                    if(r.getMask4CV() == k)
                        m_testSet.add(r);
                    else
                        m_trainSet.add(r);
                }
            }
            EM();
        }

    }
}