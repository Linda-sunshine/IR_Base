package topicmodels.embeddingModel;

import LBFGS.LBFGS;
import org.apache.commons.math3.linear.LUDecomposition;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import structures.*;
import utils.Utils;

import java.util.*;

/**
 * @author Lu Lin
 * Variational inference for Explainable Topic-Based Item Recommendation (ETBIR) model
 */
public class ETBIR {

    protected int m_varMaxIter;
    protected double m_varConverge;
    protected int m_emMaxIter;
    protected double m_emConverge;

    public int vocabulary_size;
    public int number_of_topics;
    public int number_of_users;
    public int number_of_items;

    public _User[] m_users;
    public HashMap<String, Integer> m_usersIndex; //(userID, index in m_users)
    public _Product[] m_items;
    public HashMap<String, Integer> m_itemsIndex; //(itemID, index in m_items)
    public _Corpus m_corpus;
    public HashMap<String, Integer> m_reviewIndex; //(itemIndex_userIndex, index in m_corpus.m_collection)

    public double m_rho;
    public double m_sigma;
    public double[][] m_beta; //topic_term_probability
    public double[] m_alpha;
    public double[] m_alphaG;
    public double[] m_alphaH;

    public double dAlpha;
    public double dSigma;
    public double dRho;
    public double dBeta;

    public double[] m_etaStats;
    public double[][] m_word_topic_stats;
    public double m_pStats;
    public double m_thetaStats;
    public double m_eta_p_Stats;
    public double m_eta_mean_Stats;

    public ETBIR(int emMaxIter, double emConverge, int varMaxIter, double varConverge, //stop criterion
                 int nTopics, _Corpus corpus, //user pre-defined arguments
                 double dalpha, double dsigma, double drho, double dbeta) {
        this.m_emMaxIter = emMaxIter;
        this.m_emConverge = emConverge;
        this.m_varMaxIter = varMaxIter;
        this.m_varConverge = varConverge;

        this.number_of_topics = nTopics;
        this.m_corpus = corpus;

        this.dAlpha = dalpha;
        this.dSigma = dsigma;
        this.dRho = drho;
        this.dBeta = dbeta;
    }

    public void loadCorpus(){
        System.out.println("Loading data to model...");

        m_usersIndex = new HashMap<String, Integer>();
        m_itemsIndex = new HashMap<String, Integer>();
        m_reviewIndex = new HashMap<String, Integer>();

        int u_index = 0, i_index = 0;
        for(_Doc d : m_corpus.getCollection()){
            String userID = d.getTitle();
            String itemID = d.getItemID();

            if(!m_usersIndex.containsKey(userID)){
                m_usersIndex.put(userID, u_index++);
            }

            if(!m_itemsIndex.containsKey(itemID)){
                m_itemsIndex.put(itemID, i_index++);
            }
        }
        m_users = new _User[m_usersIndex.size()];
        for(Map.Entry<String, Integer> entry: m_usersIndex.entrySet()){
            m_users[entry.getValue()] = new _User(entry.getKey());
        }

        m_items = new _Product[m_itemsIndex.size()];
        for(Map.Entry<String, Integer> entry: m_itemsIndex.entrySet()){
            m_items[entry.getValue()] = new _Product(entry.getKey());
        }

        this.number_of_items = m_items.length;
        this.number_of_users = m_users.length;
        this.vocabulary_size = m_corpus.getFeatureSize();

        for(int d = 0; d < m_corpus.getCollection().size(); d++){
            _Doc doc = m_corpus.getCollection().get(d);
            int uIndex = m_usersIndex.get(doc.getTitle());
            int iIndex = m_itemsIndex.get(doc.getItemID());
            m_reviewIndex.put(iIndex + "_" + uIndex, d);
        }

        System.out.println("-- vocabulary size: " + vocabulary_size);
        System.out.println("-- corpus size: " + m_reviewIndex.size());
        System.out.println("-- item number: " + number_of_items);
        System.out.println("-- user number: " + number_of_users);
    }

    //create space; initial parameters
    public void initModel(){
        this.m_alpha = new double[number_of_topics];
        this.m_beta = new double[number_of_topics][vocabulary_size];
        this.m_alphaG = new double[number_of_topics];
        this.m_alphaH = new double[number_of_topics];

        //initialize parameters
        Random r = new Random();
        m_rho = dRho;
        m_sigma = dSigma;
        Arrays.fill(m_alpha, dAlpha);
        double val = 0.0;
        for(int k = 0; k < number_of_topics; k++){
            double sum = 0.0;
            for(int v = 0; v < vocabulary_size; v++){
                val = r.nextDouble() + dBeta;
                sum += val;
                m_beta[k][v] = val;
            }

            for(int v = 0; v < vocabulary_size; v++){
                m_beta[k][v] = Math.log(m_beta[k][v]) - Math.log(sum);
            }
        }

        this.m_etaStats = new double[number_of_topics];
        this.m_word_topic_stats = new double[number_of_topics][vocabulary_size];
    }

    public void initDoc(_Doc doc){
        doc.m_zeta = 1.0;
        doc.m_mu = new double[number_of_topics];
        doc.m_Sigma = new double[number_of_topics];
        doc.m_phi = new double[doc.getSparse().length][number_of_topics];
        Arrays.fill(doc.m_mu, 1);
        Arrays.fill(doc.m_Sigma, 0.1);
        for(int i=0;i < doc.getSparse().length;i++){
            Arrays.fill(doc.m_phi[i], 1.0/number_of_topics);
        }
    }

    public void initUser(_User user){
        user.m_nuP = new double[number_of_topics][number_of_topics];
        user.m_SigmaP = new double[number_of_topics][number_of_topics][number_of_topics];
        for(int k = 0; k < number_of_topics; k++){
            Arrays.fill(user.m_nuP[k], 1.0);
            for(int l = 0; l < number_of_topics; l++){
                Arrays.fill(user.m_SigmaP[k][l], 0.1);
                user.m_SigmaP[k][l][l] = 0.1;
            }
        }

    }

    public void initItem(_Product item){
        item.m_eta = new double[number_of_topics];
        Arrays.fill(item.m_eta, dAlpha);

    }

    protected void initStats(){
        Arrays.fill(m_etaStats, 0.0);
        for(int k = 0; k < number_of_topics; k++){
            Arrays.fill(m_word_topic_stats[k], 0);
        }
        m_pStats = 0.0;
        m_thetaStats = 0.0;
        m_eta_p_Stats = 0.0;
        m_eta_mean_Stats = 0.0;
    }

    protected void updateStatsForItem(_Product item){
        for(int k = 0; k < number_of_topics;k++){
            m_etaStats[k] += Utils.digamma(item.m_eta[k]) - Utils.digamma(Utils.sumOfArray(item.m_eta));
        }
    }

    protected void updateStatsForUser(_User user){
        for(int k = 0; k < number_of_topics; k++){
            for(int l = 0; l < number_of_topics; l++){
                m_pStats += user.m_SigmaP[k][l][l] + user.m_nuP[k][l] * user.m_nuP[k][l];
            }
        }
    }

    protected void updateStatsForDoc(_Doc doc){
        // update m_word_topic_stats for updating beta
        double delta = 1e-6;
        _SparseFeature[] fv = doc.getSparse();
        for(int k = 0; k < number_of_topics; k++){
            for(int n = 0; n < fv.length; n++){
                int wid = fv[n].getIndex();
                double v = fv[n].getValue();
                m_word_topic_stats[k][wid] += v * doc.m_phi[n][k];
            }
        }

        // update m_thetaStats for updating rho
        for(int k = 0; k < number_of_topics; k++){
            m_thetaStats += doc.m_Sigma[k] + doc.m_mu[k] * doc.m_mu[k];
        }

        // update m_eta_p_stats for updating rho
        // update m_eta_mean_stats for updating rho
        _Product item = m_items[m_itemsIndex.get(doc.getItemID())];
        _User user = m_users[m_usersIndex.get(doc.getTitle())];
        for (int k = 0; k < number_of_topics; k++) {
            for (int l = 0; l < number_of_topics; l++) {
                m_eta_mean_Stats += item.m_eta[l] * user.m_nuP[k][l] * doc.m_mu[k];

                for (int j = 0; j < number_of_topics; j++) {
                    double term1 = user.m_SigmaP[k][l][j] + user.m_nuP[k][l] * user.m_nuP[k][j];
                    m_eta_p_Stats += item.m_eta[l] * item.m_eta[j] * term1;
                    if (j == l) {
                        term1 = user.m_SigmaP[k][l][j] + user.m_nuP[k][l] * user.m_nuP[k][j];
                        m_eta_p_Stats += item.m_eta[l] * term1;
                    }
                }
            }
        }
        double eta0 = Utils.sumOfArray(item.m_eta);
        m_eta_mean_Stats /= eta0;
        m_eta_p_Stats /= eta0 * (eta0 + 1.0);
    }

    protected double E_step(){

        int iter = 0;
        double totalLikelihood = 0.0, last = -1.0, converge = 0.0;

        do {
            initStats();
            totalLikelihood = 0.0;
            for (int i = 0; i < m_corpus.getCollection().size(); i++) {
                _Doc doc = m_corpus.getCollection().get(i);
//                System.out.println("***************** doc " + i + " ****************");
                String userID = doc.getTitle();
                String itemID = doc.getItemID();
                _User currentU = m_users[m_usersIndex.get(userID)];
                _Product currentI = m_items[m_itemsIndex.get(itemID)];

                double cur = varInferencePerDoc(doc, currentU, currentI);
                totalLikelihood += cur;
                updateStatsForDoc(doc);
            }

            for (int i = 0; i < m_users.length; i++) {
//                System.out.println("***************** user " + i + " ****************");
                _User user = m_users[i];

                double cur = varInferencePerUser(user);
                totalLikelihood += cur;
                updateStatsForUser(user);
            }

            for (int i = 0; i < m_items.length; i++) {
//                System.out.println("***************** item " + i + " ****************");
                _Product item = m_items[i];

                double cur = varInferencePerItem(item);
                totalLikelihood += cur;
                updateStatsForItem(item);
            }

            if(iter > 0) {
                converge = Math.abs((totalLikelihood - last) / last);
            }else{
                converge = 1.0;
            }
            last = totalLikelihood;
            if(converge < m_varConverge){
                break;
            }
        }while(iter++ < m_varMaxIter);

        return totalLikelihood;
    }

    protected double varInferencePerUser(_User u){
        double current = 0.0, last = 1.0, converge = 0.0;
        int iter = 0;

        do{
            update_SigmaP(u);
            update_nu(u);

            current = calc_log_likelihood_per_user(u);
            if(iter > 0){
                converge = (last - current) / last;
            }else{
                converge = 1.0;
            }
            last = current;
//            System.out.println("-- varInferencePerUser cur: " + current + "; converge: " + converge);
        } while(++iter < m_varMaxIter && Math.abs(converge) > m_varConverge);

        return current;
    }

    protected double varInferencePerItem(_Product i){
        double current = 0.0, last = 1.0, converge = 0.0;
        int iter = 0;

        do{
            update_eta(i);

            current = calc_log_likelihood_per_item(i);
            if (iter > 0){
                converge = (last - current) / last;
            }else{
                converge = 1.0;
            }
            last = current;
//            System.out.println("-- varInferencePerItem cur: " + current + "; converge: " + converge);
        } while (++iter < m_varMaxIter && Math.abs(converge) > m_varConverge);

        return current;
    }

    protected double varInferencePerDoc(_Doc d, _User u, _Product i) {
        double current = 0.0, last = 1.0, converge = 0.0;
        int iter = 0;

        do {
            update_phi(d);
            update_zeta(d);
            update_mu(d, u ,i);
            update_zeta(d);
            update_SigmaTheta(d);
            update_zeta(d);

            current = calc_log_likelihood_per_doc(d, u, i);
            if (iter > 0) {
                converge = (last-current) / last;
            }else{
                converge = 1.0;
            }
            last = current;
//            System.out.println("-- varInferencePerDoc cur: " + current + "; converge: " + converge);
        } while (++iter < m_varMaxIter && Math.abs(converge) > m_varConverge);

        return current;
    }

    //variational inference for p(z|w,\phi) for each document
    public void update_phi(_Doc d){
        double logSum, v;
        int wid;
        _SparseFeature[] fv = d.getSparse();

        for (int n = 0; n < fv.length; n++) {
            wid = fv[n].getIndex();
            v = fv[n].getValue();
            for (int k = 0; k < number_of_topics; k++) {
                d.m_phi[n][k] = m_beta[k][wid] + d.m_mu[k];
            }
            // normalize
            logSum = Utils.logSum(d.m_phi[n]);
            for (int k = 0; k < number_of_topics; k++) {
                d.m_phi[n][k] = Math.exp(d.m_phi[n][k] - logSum);
            }
        }
    }

    //variational inference for p(\theta|\mu,\Sigma) for each document
    public void update_zeta(_Doc d){
        //estimate zeta
        d.m_zeta = 0;
        for (int k = 0; k < number_of_topics; k++) {
            d.m_zeta += Math.exp(d.m_mu[k] + 0.5 * d.m_Sigma[k]);
        }

    }

    // alternative: line search / fixed-stepsize gradient descent
    public void update_mu(_Doc doc, _User user, _Product item){
        double fValue = 1.0, lastFValue = 1.0, cvg = 1e-4, diff, iterMax = 60, iter = 0;
        double last = 1.0;
        double cur = 0.0;
        double stepsize = 1e-1, alpha = 1e-4, c2 = 0.5, beta = 0.8;
        double[] m_diagG = new double[number_of_topics];
        double[] m_muG = new double[number_of_topics]; // gradient for mu
        double[] mu_diag = new double[number_of_topics];
        double muG2Norm = 0.0, diagG2Norm = 0.0;
        int N = doc.getTotalDocLength();

        double[] m_phiStat = new double[number_of_topics];
        Arrays.fill(m_phiStat, 0.0);
        _SparseFeature[] fv = doc.getSparse();
        for(int k = 0;k < number_of_topics; k++) {
            for (int n = 0; n < fv.length; n++) {
                int wid = fv[n].getIndex();
                double v = fv[n].getValue();
                m_phiStat[k] += v * doc.m_phi[n][k];
            }
        }

        Arrays.fill(mu_diag, 0.0);

        double moment, zeta_stat = 1.0 / doc.m_zeta;
        do {
            //update gradient of mu
            last = 0.0;
            for (int k = 0; k < number_of_topics; k++) {
                moment = Math.exp(doc.m_mu[k] + 0.5 * doc.m_Sigma[k]);
                m_muG[k] = -(-m_rho * (doc.m_mu[k] - Utils.dotProduct(item.m_eta, user.m_nuP[k]) / Utils.sumOfArray(item.m_eta))
                        + m_phiStat[k] - N * zeta_stat * moment);//-1 because LBFGS is minimization
                last += -(-0.5 * m_rho * (doc.m_mu[k] * doc.m_mu[k]
                        - 2 * doc.m_mu[k] * Utils.dotProduct(item.m_eta, user.m_nuP[k]) / Utils.sumOfArray(item.m_eta))
                        + doc.m_mu[k] * m_phiStat[k] - N * zeta_stat * moment);
            }

            stepsize = 1e-2;
            int iterLS = 10, i=0;
            //line search
//            do{
//                stepsize = beta * stepsize;
//                cur = 0.0;
//                for(int k=0;k < number_of_topics;k++) {
//                    mu_diag[k] = doc.m_mu[k] - stepsize * m_muG[k];
//                    moment = Math.exp(mu_diag[k] + 0.5 * mu_diag[k]);
//                    m_diagG[k] = -(-m_rho * (mu_diag[k] - Utils.dotProduct(item.m_eta, user.m_nuP[k]) / Utils.sumOfArray(item.m_eta))
//                            + m_phiStat[k] - N * zeta_stat * moment);
//                    cur += -(-0.5 * m_rho * (mu_diag[k] * mu_diag[k]
//                            - 2 * mu_diag[k] * Utils.dotProduct(item.m_eta, user.m_nuP[k]) / Utils.sumOfArray(item.m_eta))
//                            + mu_diag[k] * m_phiStat[k] - N * zeta_stat * moment);
//                }
//                diff = cur - last;
//                muG2Norm = Utils.dotProduct(m_muG, m_muG);
//                diagG2Norm = Utils.dotProduct(m_diagG, m_diagG);
//                i += 1;
//                System.out.println("---- line search for mu: cur: "
//                        + cur + "; diff: " + diff + "; gradient: " + diagG2Norm + "; stepsize: " + stepsize);
//                if(i > iterLS && diff <= 0.0){
//                    break;
//                }
//            }while(diff > - alpha * stepsize * muG2Norm || muG2Norm > diagG2Norm);

            //fix stepsize
            lastFValue = fValue;
            fValue = 0.0;
            for(int k=0;k < number_of_topics;k++) {
                doc.m_mu[k] = doc.m_mu[k] - stepsize * m_muG[k];
                moment = Math.exp(doc.m_mu[k] + 0.5 * doc.m_Sigma[k]);
                fValue += -(-0.5 * m_rho * (doc.m_mu[k] * doc.m_mu[k]
                        - 2 * doc.m_mu[k] * Utils.dotProduct(item.m_eta, user.m_nuP[k]) / Utils.sumOfArray(item.m_eta))
                        + doc.m_mu[k] * m_phiStat[k] - N * zeta_stat * moment);
            }
//            LBFGS.lbfgs(number_of_topics,4, doc.m_mu, fValue, m_muG,false, mu_diag, iprint, 1e-6, 1e-16, iflag);
            diff = (lastFValue - fValue) / lastFValue;
//            if(iter % 5 == 0) {
//                System.out.println("----- update mu cur: " + fValue + "; diff: " + diff
//                        + "; gradient: " + Utils.dotProduct(m_muG, m_muG));
//            }
        } while (iter++ < iterMax && Math.abs(diff) > cvg);
    }

    public void update_SigmaTheta(_Doc d){
        double fValue = 1.0, lastFValue = 1.0, cvg = 1e-6, diff, iterMax = 20, iter = 0;
        double last = 1.0;
        double cur = 0.0;
        double stepsize , alpha = 0.5, beta = 0.8;
        int N = d.getTotalDocLength();
        double[] m_SigmaG = new double[number_of_topics]; // gradient for Sigma
        double[] m_SigmaGl = new double[number_of_topics];
        double[] m_sigmaNew = new double[number_of_topics];
        Arrays.fill(m_sigmaNew, 0.0);

        double[] m_sigmaSqrt = new double[number_of_topics];
        for(int k=0; k < number_of_topics; k++){
            m_sigmaSqrt[k] = Math.sqrt(d.m_Sigma[k]);
        }

        double moment, sigma;
        double sigma2Norm=0.0, diag2Norm = 0.0;
        do {
            //update gradient of sigma
            last = 0.0;
            for (int k = 0; k < number_of_topics; k++) {
                sigma = Math.pow(m_sigmaSqrt[k], 2);
                moment = Math.exp(d.m_mu[k] + 0.5 * sigma);
                m_SigmaG[k] = -(-m_rho * m_sigmaSqrt[k] - N * m_sigmaSqrt[k] * moment / d.m_zeta + 1.0 / m_sigmaSqrt[k]); //-1 because LBFGS is minimization
                last += -(-0.5 * m_rho * sigma - N * moment / d.m_zeta + 0.5 * Math.log(sigma));
            }

            //line search
            stepsize = 1e-2;
//            int iterLS = 10, i=0;
//            do{
//                cur = 0.0;
//                stepsize = beta * stepsize;
//                for (int k = 0; k < number_of_topics; k++) {
//                    m_sigmaNew[k] = m_sigmaSqrt[k] - stepsize * m_SigmaG[k];
//                    sigma = Math.pow(m_sigmaNew[k], 2);
//                    moment = Math.exp(d.m_mu[k] + 0.5 * sigma);
//                    m_SigmaGl[k] = -(-m_rho * m_sigmaNew[k] - N * moment * m_sigmaNew[k] / d.m_zeta + 1.0 / m_sigmaNew[k]);
//                    cur += -(-0.5 * m_rho * sigma - N * moment / d.m_zeta + 0.5 * Math.log(sigma));
//                }
//                diff = cur - last;
//                sigma2Norm = Utils.dotProduct(m_SigmaG, m_SigmaG);
//                diag2Norm = Utils.dotProduct(m_SigmaGl, m_SigmaGl);
//                i += 1;
////                System.out.println("---- line search for sigmaTheta: cur: "
////                        + cur + "; diff: " + diff + "; gradient: " + diag2Norm + "; stepsize: " + stepsize);
//                if(i > iterLS && diff <= 0.0){
//                    break;
//                }
//            }while(diff > - alpha * stepsize * sigma2Norm && diag2Norm > sigma2Norm);

            lastFValue = fValue;
            fValue = 0.0;
            for(int k = 0; k < number_of_topics;k ++) {
                m_sigmaSqrt[k] = m_sigmaSqrt[k] - stepsize * m_SigmaG[k];
                sigma = Math.pow(m_sigmaSqrt[k], 2);
                moment = Math.exp(d.m_mu[k] + 0.5 * sigma);
                fValue += -(-0.5 * m_rho * sigma - N * moment / d.m_zeta + 0.5 * Math.log(sigma));
            }

//            LBFGS.lbfgs(number_of_topics,4, d.m_Sigma, fValue, m_SigmaG,false, sigma_diag, iprint, 1e-6, 1e-32, iflag);

            diff = (lastFValue - fValue) / lastFValue;
//            System.out.println("---- update thetaSigma cur: " + fValue + "; diff: " + diff + "; gradient: "
//                    + Utils.dotProduct(m_SigmaG, m_SigmaG));
        } while(iter++ < iterMax && Math.abs(diff) > cvg);

        for(int k=0; k < number_of_topics; k++){
            d.m_Sigma[k] = Math.pow(m_sigmaSqrt[k], 2);
        }
//        System.out.println("sigmasum: " + Utils.sumOfArray(d.m_Sigma));

    }

    //variational inference for p(P|\nu,\Sigma) for each user
    public void update_SigmaP(_User u){
        RealMatrix eta_stat_sigma = MatrixUtils.createRealIdentityMatrix(number_of_topics).scalarMultiply(m_sigma);
        for (int item_i = 0; item_i < number_of_items; item_i++) {
            RealMatrix eta_vec = MatrixUtils.createColumnRealMatrix(m_items[item_i].m_eta);
            double eta_0 = Utils.sumOfArray(m_items[item_i].m_eta);
            RealMatrix eta_stat_i = MatrixUtils.createRealDiagonalMatrix(m_items[item_i].m_eta).add(eta_vec.multiply(eta_vec.transpose()));
            eta_stat_sigma = eta_stat_sigma.add(eta_stat_i.scalarMultiply(m_rho / (eta_0 * (eta_0 + 1.0))));
        }
//        System.out.println("-- sigmaP before inverse: " + Arrays.toString(eta_stat_sigma.getColumn(1)));
        eta_stat_sigma = new LUDecomposition(eta_stat_sigma).getSolver().getInverse();
//        System.out.println("-- update sigmaP: " + Arrays.toString(eta_stat_sigma.getColumn(1)));
        for (int k = 0; k < number_of_topics; k++) {
            u.m_SigmaP[k] = eta_stat_sigma.getData();
        }
//        System.out.println("-- update sigmaP: now: " + Arrays.toString(u.m_SigmaP[0][0]));
    }

    //variational inference for p(P|\nu,\Sigma) for each user
    public void update_nu(_User u){
        RealMatrix eta_stat_sigma = MatrixUtils.createRealMatrix(u.m_SigmaP[0]);

//        System.out.println("-- update nuP: origin: " + Arrays.toString(u.m_nuP[0]));
        for (int k = 0; k < number_of_topics; k++) {
            RealMatrix eta_stat_nu = MatrixUtils.createColumnRealMatrix(new double[number_of_topics]);
            for (int item_i = 0; item_i < number_of_items; item_i++) {
                RealMatrix eta_vec = MatrixUtils.createColumnRealMatrix(m_items[item_i].m_eta);
                double eta_0 = Utils.sumOfArray(m_items[item_i].m_eta);
                _Doc d = m_corpus.getCollection().get(m_reviewIndex.get(item_i + "_"
                        + m_usersIndex.get(u.getUserID())));
                eta_stat_nu = eta_stat_nu.add(eta_vec.scalarMultiply(d.m_mu[k] / eta_0));
            }
            u.m_nuP[k] = eta_stat_sigma.multiply(eta_stat_nu).scalarMultiply(m_rho).getColumn(0);
        }
//        System.out.println("-- update nuP: origin: " + Arrays.toString(u.m_nuP[0]));
    }

    public void update_eta(_Product i){
        double fValue = 1.0, lastFValue = 1.0, cvg = 1e-6, diff, iterMax = 20, iter = 0;
        double last = 1.0;
        double cur = 0.0, check = 0;
        double stepsize = 1e-1, alpha = 0.5, beta = 0.8;
        double[] m_etaG = new double[number_of_topics];
        double[] eta_diag = new double[number_of_topics];
        for(int k = 0; k < number_of_topics; k++){
            eta_diag[k] = i.m_eta[k];
        }

        double eta_0, monitorNeg = 0.0;
        do{
            lastFValue = fValue;
            last = 0.0;
            monitorNeg = 0.0;
            stepsize = 1e-3;
            for(int k = 0; k < number_of_topics; k++) {

                //might be optimized using global stats
                eta_0 = Utils.sumOfArray(i.m_eta);
                double gTerm1 = 0.0;
                double gTerm2 = 0.0;
                double gTerm3 = 0.0;
                double gTerm4 = 0.0;
                double term1 = 0.0;
                double term2 = 0.0;
                for (int uid = 0; uid < number_of_users; uid++) {
                    _Doc d = m_corpus.getCollection().get(m_reviewIndex.get(
                            m_itemsIndex.get(i.getID()) + "_" + uid));
                    for (int j = 0; j < number_of_topics; j++) {
                        gTerm1 += m_users[uid].m_nuP[j][k] * d.m_mu[j];
                        term1 += i.m_eta[k] * m_users[uid].m_nuP[j][k] * d.m_mu[j];

                        for (int l = 0; l < number_of_topics; l++) {
                            gTerm2 += i.m_eta[l] * m_users[uid].m_nuP[j][l] * d.m_mu[j];

                            gTerm3 += i.m_eta[l] * (m_users[uid].m_SigmaP[j][l][k] + m_users[uid].m_nuP[j][l] * m_users[uid].m_nuP[j][k]);
                            if (l == k) {
                                gTerm3 += (i.m_eta[l] + 1.0) * (m_users[uid].m_SigmaP[j][l][k] + m_users[uid].m_nuP[j][l] * m_users[uid].m_nuP[j][k]);
                            }

                            term2 += i.m_eta[l] * i.m_eta[k] * (m_users[uid].m_SigmaP[j][l][k] + m_users[uid].m_nuP[j][l] * m_users[uid].m_nuP[j][k]);
                            if (l == k) {
                                term2 += i.m_eta[k] * (m_users[uid].m_SigmaP[j][k][k] + m_users[uid].m_nuP[j][k] * m_users[uid].m_nuP[j][k]);
                            }

                            for (int p = 0; p < number_of_topics; p++) {
                                gTerm4 += i.m_eta[l] * i.m_eta[p] * (m_users[uid].m_SigmaP[j][l][p] + m_users[uid].m_nuP[j][l] * m_users[uid].m_nuP[j][p]);
                                if (p == l) {
                                    gTerm4 += i.m_eta[p] * (m_users[uid].m_SigmaP[j][l][p] + m_users[uid].m_nuP[j][l] * m_users[uid].m_nuP[j][p]);
                                }
                            }
                        }
                    }
                }
                m_etaG[k] = -(Utils.trigamma(i.m_eta[k]) * (m_alpha[k] - i.m_eta[k])
                        - Utils.trigamma(eta_0) * (Utils.sumOfArray(m_alpha) - eta_0)
                        + m_rho * gTerm1 / eta_0 - m_rho * gTerm2 / (Math.pow(eta_0, 2))
                        - m_rho * gTerm3 / (2 * eta_0 * (eta_0 + 1.0))
                        + m_rho * (2 * eta_0 + 1.0) * gTerm4 / (2 * Math.pow(eta_0, 2) * Math.pow(eta_0 + 1.0, 2)));
                if(k == 0) {
                    double eps = 1e-6;
                    i.m_eta[k] = i.m_eta[k] + eps;
                    double post = -((m_alpha[k] - i.m_eta[k]) * (Utils.digamma(i.m_eta[k]) - Utils.digamma(eta_0))
                            - Utils.lgamma(eta_0) + Utils.lgamma(i.m_eta[k])
                            + m_rho * term1 / eta_0 - m_rho * term2 / (2 * eta_0 * (eta_0 + 1.0)));
                    i.m_eta[k] = i.m_eta[k] - eps;
                    double pre = -((m_alpha[k] - i.m_eta[k]) * (Utils.digamma(i.m_eta[k]) - Utils.digamma(eta_0))
                            - Utils.lgamma(eta_0) + Utils.lgamma(i.m_eta[k])
                            + m_rho * term1 / eta_0 - m_rho * term2 / (2 * eta_0 * (eta_0 + 1.0)));
                    check = (post - pre) / eps;
                }


                last += -((m_alpha[k] - i.m_eta[k]) * (Utils.digamma(i.m_eta[k]) - Utils.digamma(eta_0))
                        - Utils.lgamma(eta_0) + Utils.lgamma(i.m_eta[k])
                        + m_rho * term1 / eta_0 - m_rho * term2 / (2 * eta_0 * (eta_0 + 1.0)));
            }
//            do{
//                stepsize = beta * stepsize;
//                for(int k = 0; k < number_of_topics; k++) {
//                    eta_diag[k] = i.m_eta[k] - stepsize * m_etaG[k];
//                }
//                double eta0_diag = Utils.sumOfArray(eta_diag);
//                double term1_diag = 0.0;
//                double term2_diag = 0.0;
//                cur = 0.0;
//                for(int k = 0; k < number_of_topics; k++) {
//                    for (int uid = 0; uid < number_of_users; uid++) {
//                        _Doc d = m_corpus.getCollection().get(m_reviewIndex.get(
//                                m_itemsIndex.get(i.getID()) + "_" + uid));
//                        for (int j = 0; j < number_of_topics; j++) {
//                            term1_diag += eta_diag[k] * m_users[uid].m_nuP[j][k] * d.m_mu[j];
//                            for (int l = 0; l < number_of_topics; l++) {
//                                term2_diag += eta_diag[l] * eta_diag[k] * (m_users[uid].m_SigmaP[j][l][k] + m_users[uid].m_nuP[j][l] * m_users[uid].m_nuP[j][k]);
//                                if (l == k) {
//                                    term2_diag += eta_diag[k] * (m_users[uid].m_SigmaP[j][k][k] + m_users[uid].m_nuP[j][k] * m_users[uid].m_nuP[j][k]);
//                                }
//                            }
//                        }
//                    }
//                    cur += -((m_alpha[k] - eta_diag[k]) * (Utils.digamma(eta_diag[k]) - Utils.digamma(eta0_diag))
//                            - Utils.lgamma(eta0_diag) + Utils.lgamma(eta_diag[k])
//                            + m_rho * term1_diag / eta0_diag - m_rho * term2_diag / (2 * eta0_diag * (eta0_diag + 1.0)));
//                }
//                diff = cur - last;
////                System.out.println("----  line search: cur: " + cur + "; diff: " + diff
////                            + "; eta_diag: " + eta_diag[0]
////                            + "; etaG: " + Utils.dotProduct(m_etaG, m_etaG) + "; eta0_diag: " + eta0_diag
////                            + "; stepsize: " + stepsize);
//
//            }while(diff > - alpha * stepsize * Utils.dotProduct(m_etaG, m_etaG));
            // fix stepsize
            for(int k = 0; k < number_of_topics; k++) {
                i.m_eta[k] = i.m_eta[k] - stepsize * m_etaG[k];
                if (i.m_eta[k] <= 0) {
                    monitorNeg += i.m_eta[k];
                }
            }
            double eta0_diag = Utils.sumOfArray(i.m_eta);
            for(int k = 0; k < number_of_topics; k++) {
                double term1_diag = 0.0;
                double term2_diag = 0.0;
                for (int uid = 0; uid < number_of_users; uid++) {
                    _Doc d = m_corpus.getCollection().get(m_reviewIndex.get(
                            m_itemsIndex.get(i.getID()) + "_" + uid));
                    for (int j = 0; j < number_of_topics; j++) {
                        term1_diag += i.m_eta[k] * m_users[uid].m_nuP[j][k] * d.m_mu[j];
                        for (int l = 0; l < number_of_topics; l++) {
                            term2_diag += i.m_eta[l] * i.m_eta[k] * (m_users[uid].m_SigmaP[j][l][k] + m_users[uid].m_nuP[j][l] * m_users[uid].m_nuP[j][k]);
                            if (l == k) {
                                term2_diag += i.m_eta[k] * (m_users[uid].m_SigmaP[j][k][k] + m_users[uid].m_nuP[j][k] * m_users[uid].m_nuP[j][k]);
                            }
                        }
                    }
                }
                cur += -((m_alpha[k] - i.m_eta[k]) * (Utils.digamma(i.m_eta[k]) - Utils.digamma(eta0_diag))
                        - Utils.lgamma(eta0_diag) + Utils.lgamma(i.m_eta[k])
                        + m_rho * term1_diag / eta0_diag - m_rho * term2_diag / (2 * eta0_diag * (eta0_diag + 1.0)));
            }

            fValue = cur;
            diff = (lastFValue - fValue) / lastFValue;
            System.out.println("----- update eta cur: " + cur + "; diff: " + diff + "; gradient: " + m_etaG[0]
                    + "; gradient check: " + check + "; nonnegative: " + monitorNeg);

//            System.out.println("-- update eta: fValue: " + fValue + "; diff: " + diff
//                    + "; gradient: " + Utils.dotProduct(m_etaG, m_etaG) + "; eta: " + Utils.sumOfArray(i.m_eta)
//                    + "; monitor: " + monitorNeg);
//            LBFGS.lbfgs(number_of_topics,4, i.m_eta, fValue, m_etaG,false, eta_diag, iprint, 1e-6, 1e-32, iflag);
        }while(iter++ < iterMax && Math.abs(diff) > cvg);
    }

    public void M_step() {
        //maximize likelihood for \rho of p(\theta|P\gamma, \rho)
        m_rho = number_of_topics / (m_thetaStats + m_eta_p_Stats - 2 * m_eta_mean_Stats);

        //maximize likelihood for \sigma
        m_sigma = number_of_topics / m_pStats;

        //maximize likelihood for \beta
        for(int k = 0 ;k < number_of_topics; k++){
            double sum = Utils.sumOfArray(m_word_topic_stats[k]);
            for(int v = 0; v < vocabulary_size; v++){
                m_beta[k][v] = Math.log(m_word_topic_stats[k][v]) - Math.log(sum);
//                m_beta[k][v] = m_word_topic_stats[k][v] / sum;
            }
        }

        //maximize likelihood for \alpha using Newton
        int i = 0;
        double diff = 0.0, alphaSum, diAlphaSum, z, c1, c2, c, deltaAlpha;
        do{
            alphaSum = Utils.sumOfArray(m_alpha);
            diAlphaSum = Utils.digamma(alphaSum);
            z = number_of_items * Utils.trigamma(alphaSum);

            c1 = 0; c2 = 0;
            for(int k = 0; k < number_of_topics; k++){
                m_alphaG[k] = number_of_items * (diAlphaSum - Utils.digamma(m_alpha[k])) + m_etaStats[k];
                m_alphaH[k] = - number_of_items * Utils.trigamma(m_alpha[k]);

                c1 += m_alphaG[k] / m_alphaH[k];
                c2 += 1.0 / m_alphaH[k];
            }
            c = c1 / (1.0/z + c2);

            diff = 0.0;
            for(int k = 0; k < number_of_topics; k++){
                deltaAlpha = (m_alphaG[k] -c) / m_alphaH[k];
                m_alpha[k] -= deltaAlpha;
                diff += deltaAlpha * deltaAlpha;
            }
            diff /= number_of_topics;
        }while(++i < m_varMaxIter && diff > m_varConverge);

    }

    // calculate the likelihood of user-related terms (term2-term7)
    protected double calc_log_likelihood_per_user(_User u){
        double log_likelihood = 0.0;

        for(int k = 0; k < number_of_topics; k++){
            double temp1 = 0.0;
            for(int l = 0; l < number_of_topics; l++) {
                temp1 += u.m_SigmaP[k][l][l] + u.m_nuP[k][l] * u.m_nuP[k][l];
            }
            double det = new LUDecomposition(MatrixUtils.createRealMatrix(u.m_SigmaP[k])).getDeterminant();
            log_likelihood += -0.5 * (temp1 * m_sigma - number_of_topics)
                    + 0.5 * (number_of_topics * Math.log(m_sigma) + Math.log(det));
        }

        return log_likelihood;
    }

    // calculate the likelihood of item-related terms (term1-term6)
    protected double calc_log_likelihood_per_item(_Product i){
        double log_likelihood = 0.0;
        double eta0 = Utils.sumOfArray(i.m_eta);
        double diGammaEtaSum = Utils.digamma(eta0);
        double lgammaEtaSum = Utils.lgamma(eta0);
        double lgammaAlphaSum = Utils.lgamma(Utils.sumOfArray(m_alpha));

        for(int k = 0; k < number_of_topics; k++){
            log_likelihood += (m_alpha[k] - i.m_eta[k]) * (Utils.digamma(i.m_eta[k]) - diGammaEtaSum);
            log_likelihood -= Utils.lgamma(m_alpha[k]) - Utils.lgamma(i.m_eta[k]);
        }
        log_likelihood += lgammaAlphaSum - lgammaEtaSum;
        log_likelihood += log_likelihood;

        return log_likelihood;
    }

    // calculate the likelihood of doc-related terms (term3-term8 + term4-term9 + term5)
    protected double calc_log_likelihood_per_doc(_Doc doc, _User currentU, _Product currentI) {

        double log_likelihood = 0.0;
        double eta0 = Utils.sumOfArray(currentI.m_eta);

        // (term3-term8)
        double term1 = 0.0;
        double term2 = 0.0;
        double term3 = 0.0;
        double term4 = 0.0;
        double part3 = 0.0;
        for(int k = 0; k < number_of_topics; k++){
            term1 += doc.m_Sigma[k] + doc.m_mu[k] * doc.m_mu[k];
            for(int j = 0; j < number_of_topics; j++){
                term2 += currentI.m_eta[k] * currentU.m_nuP[j][k] * doc.m_mu[j];

                for(int l = 0; l < number_of_topics; l++){
                    term3 += currentI.m_eta[j] * currentI.m_eta[l] *
                            (currentU.m_SigmaP[k][j][l] + currentU.m_nuP[k][j] * currentU.m_nuP[k][l]);
                    if(l == j){
                        term3 += currentI.m_eta[l] *
                                (currentU.m_SigmaP[k][j][l] + currentU.m_nuP[k][j] * currentU.m_nuP[k][l]);
                    }
                }
            }
            term4 += Math.log(m_rho * doc.m_Sigma[k]);
        }
        part3 += -m_rho * (0.5 * term1 - 2 * term2 / eta0 + term3 / (eta0 * (eta0 + 1.0))) + number_of_topics/2.0
                + 0.5 * term4;
        log_likelihood += part3;

        //part4
        int wid;
        double v;
        double part4 = 0.0, part5 = 0.0;
        term1 = 0.0;
        term2 = 0.0;
        term3 = 0.0;
        _SparseFeature[] fv = doc.getSparse();
//        System.out.println("file length: " + fv.length);
        for(int k = 0; k < number_of_topics; k++) {
            for (int n = 0; n < fv.length; n++) {
                wid = fv[n].getIndex();
                v = fv[n].getValue();
                term1 += v * doc.m_phi[n][k] * doc.m_mu[k];
                term3 += v * doc.m_phi[n][k] * Math.log(doc.m_phi[n][k]);
                part5 += v * doc.m_phi[n][k] * m_beta[k][wid];
            }
            term2 += Math.exp(doc.m_mu[k] + doc.m_Sigma[k]/2.0);
        }
        part4 += term1 - term2 / doc.m_zeta + 1.0 - Math.log(doc.m_zeta) - term3;
        log_likelihood += part4;
        log_likelihood += part5;

        return log_likelihood;
    }

    public void EM(){
        System.out.println("Initializing model...");
        initModel();

        System.out.println("Initializing documents...");
        for(_Doc doc : m_corpus.getCollection()){
            initDoc(doc);
        }
        System.out.println("Initializing users...");
        for(_User user : m_users){
            initUser(user);
        }
        System.out.println("Initializing items...");
        for(_Product item : m_items){
            initItem(item);
        }

        int iter = 0;
        double lastAllLikelihood = 1.0;
        double currentAllLikelihood;
        double converge = 0.0;
        do{
            currentAllLikelihood = E_step();
            for(int k = 0; k < number_of_topics;k++){
                for(int v=0; v < vocabulary_size; v++){
                    m_word_topic_stats[k][v] += 1e-5;
                }
            }
            if(iter > 0){
                converge = (lastAllLikelihood - currentAllLikelihood) / lastAllLikelihood;
            }else{
                converge = 1.0;
            }

            if(converge < 0){
                m_varMaxIter += 10;
                System.out.println("! E_step not converge...");
            }else{
                M_step();
                lastAllLikelihood = currentAllLikelihood;
                System.out.format("%s step: likelihood is %.3f, converge to %f...\n",
                        iter, currentAllLikelihood, converge);
                iter++;
                if(converge < m_emConverge)
                    break;
            }
            System.out.println("sigma: " + m_sigma + "; rho: " + m_rho);
            System.out.println(Utils.sumOfArray(m_word_topic_stats[0]));
        }while(iter < m_emMaxIter && (converge < 0 || converge > m_emConverge));
    }

}