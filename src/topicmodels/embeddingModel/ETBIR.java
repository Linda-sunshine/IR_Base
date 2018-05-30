package topicmodels.embeddingModel;

import java.io.*;
import java.util.*;

import Analyzer.BipartiteAnalyzer;
import org.apache.commons.math3.linear.LUDecomposition;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;

import structures.*;
import topicmodels.LDA.LDA_Variational;
import topicmodels.markovmodel.HTSM;
import topicmodels.multithreads.TopicModelWorker;
import utils.Utils;
import LBFGS.LBFGS;

/**
 * @author Lu Lin
 * Variational inference for Explainable Topic-Based Item Recommendation (ETBIR) model
 */
public class ETBIR extends LDA_Variational {

    protected int number_of_users;
    protected int number_of_items;

    protected List<_User> m_users;
    protected List<_Product> m_items;

    protected HashMap<String, Integer> m_usersIndex; //(userID, index in m_users)
    protected HashMap<String, Integer> m_itemsIndex; //(itemID, index in m_items)
    protected HashMap<String, Integer> m_reviewIndex; //(itemIndex_userIndex, index in m_corpus.m_collection)

    protected HashMap<Integer, ArrayList<Integer>>  m_mapByUser; //adjacent list for user, controlled by m_testFlag.
    protected HashMap<Integer, ArrayList<Integer>> m_mapByItem;
    protected HashMap<Integer, ArrayList<Integer>> m_mapByUser_test; //test
    protected HashMap<Integer, ArrayList<Integer>> m_mapByItem_test;
    protected BipartiteAnalyzer m_bipartite;

    protected double m_rho;
    protected double m_sigma;

    protected double m_pStats;
    protected double m_thetaStats;
    protected double m_eta_p_Stats;
    protected double m_eta_mean_Stats;

    double d_mu = 1.0, d_sigma_theta = 10;
    double d_nu = 1.0, d_sigma_P = 10;

    public ETBIR(int emMaxIter, double emConverge,
                 double beta, _Corpus corpus, double lambda,
                 int number_of_topics, double alpha, int varMaxIter, double varConverge, //LDA_variational
                 double sigma, double rho) {
        super(emMaxIter, emConverge,
                beta, corpus, lambda,
                number_of_topics, alpha, varMaxIter, varConverge);

        this.m_sigma = sigma;
        this.m_rho = rho;
    }

    @Override
    public String toString(){
        return String.format("ETBIR[k:%d, alpha:%.5f, beta:%.5f, simga:%.5f,rho:%,5f, item~N(%.5f, %.5f), user~N(%.5f, %.5sf)]\n",
                number_of_topics, d_alpha, d_beta, this.m_sigma, this.m_rho, d_mu, d_sigma_theta, d_nu, d_sigma_P);
    }

    @Override
    protected void init() { // clear up for next iteration during EM
        super.init();

        m_pStats = 0.0;
        m_thetaStats = 0.0;
        m_eta_p_Stats = 0.0;
        m_eta_mean_Stats = 0.0;
    }

    protected void updateStats4Item(_Product4ETBIR item){
        double digammaSum = Utils.digamma(Utils.sumOfArray(item.m_eta));
        for(int k = 0; k < number_of_topics; k++)
            m_alphaStat[k] += Utils.digamma(item.m_eta[k]) - digammaSum;
    }

    protected void updateStats4User(_User4ETBIR user){
        for(int k = 0; k < number_of_topics; k++){
            for(int l = 0; l < number_of_topics; l++){
                m_pStats += user.m_SigmaP[k][l][l] + user.m_nuP[k][l] * user.m_nuP[k][l];
            }
        }
    }

    protected void updateStats4Doc(_Doc4ETBIR doc){
        // update m_word_topic_stats for updating beta
        _SparseFeature[] fv = doc.getSparse();
        int wid;
        double v;
        for(int n=0; n<fv.length; n++) {
            wid = fv[n].getIndex();
            v = fv[n].getValue();
            for(int i=0; i<number_of_topics; i++)
                word_topic_sstat[i][wid] += v*doc.m_phi[n][i];
        }

        // update m_thetaStats for updating rho
        for(int k = 0; k < number_of_topics; k++)
            m_thetaStats += doc.m_Sigma[k] + doc.m_mu[k] * doc.m_mu[k];

        // update m_eta_p_stats for updating rho
        // update m_eta_mean_stats for updating rho
        double eta_mean_temp = 0.0;
        double eta_p_temp = 0.0;
        _Product4ETBIR item = (_Product4ETBIR) m_items.get(m_itemsIndex.get(doc.getItemID()));
        _User4ETBIR user = (_User4ETBIR) m_users.get(m_usersIndex.get(doc.getUserID()));
        for (int k = 0; k < number_of_topics; k++) {
            for (int l = 0; l < number_of_topics; l++) {
                eta_mean_temp += item.m_eta[l] * user.m_nuP[k][l] * doc.m_mu[k];

                for (int j = 0; j < number_of_topics; j++) {
                    double term1 = user.m_SigmaP[k][l][j] + user.m_nuP[k][l] * user.m_nuP[k][j];
                    eta_p_temp += item.m_eta[l] * item.m_eta[j] * term1;
                    if (j == l) {
                        term1 = user.m_SigmaP[k][l][j] + user.m_nuP[k][l] * user.m_nuP[k][j];
                        eta_p_temp += item.m_eta[l] * term1;
                    }
                }
            }
        }

        double eta0 = Utils.sumOfArray(item.m_eta);
        m_eta_mean_Stats += eta_mean_temp / eta0;
        m_eta_p_Stats += eta_p_temp / (eta0 * (eta0 + 1.0));
    }

    // return log-likelihood
    @Override
    public double calculate_E_step(_Doc d){
        _Doc4ETBIR doc = (_Doc4ETBIR)d;

        String userID = doc.getUserID();
        String itemID = doc.getItemID();
        _User4ETBIR currentU = (_User4ETBIR) m_users.get(m_usersIndex.get(userID));
        _Product4ETBIR currentI = (_Product4ETBIR) m_items.get(m_itemsIndex.get(itemID));

        double cur = varInference4Doc(doc, currentU, currentI);
        updateStats4Doc(doc);
        return cur;
    }

    protected double E_step(){
        int iter = 0;
        double totalLikelihood = 0.0, last = -1.0, converge = 0.0;

        do {
            init();

            totalLikelihood = 0.0;
            for (_Doc d:m_trainSet) {
                totalLikelihood += calculate_E_step(d);
            }

            for (int u_idx:m_mapByUser.keySet()) {
                _User4ETBIR user = (_User4ETBIR) m_users.get(u_idx);
                totalLikelihood += varInference4User(user);
                updateStats4User(user);
            }

            for (int i_idx:m_mapByItem.keySet()) {
                _Product4ETBIR item = (_Product4ETBIR) m_items.get(i_idx);
                totalLikelihood += varInference4Item(item);
                updateStats4Item(item);
            }

            if(Double.isNaN(totalLikelihood)){
                System.out.println("! E_step produces NaN likelihood...");
                break;
            }

            if(iter > 0)
                converge = Math.abs((totalLikelihood - last) / last);
            else
                converge = 1.0;

            last = totalLikelihood;
            if(converge < m_varConverge)
                break;
            System.out.print("---likelihood: " + last + "\n");

        }while(iter++ < m_varMaxIter);
        System.out.print(String.format("Current likelihood: %.4f", totalLikelihood));
        return totalLikelihood;
    }

    public double inference(_User4ETBIR user) {
//        user.setTopics4Variational(number_of_topics, d_nu, d_sigma_P);
        double likelihood = varInference4User(user);
        return likelihood;
    }

    public double inference(_Product4ETBIR item){
//        item.setTopics4Variational(number_of_topics, d_alpha+1);
        double likelihood = varInference4Item(item);
        return likelihood;
    }

    @Override
    protected void initTestDoc(_Doc d) {
        ((_Doc4ETBIR) d).setTopics4Variational(number_of_topics, d_alpha, d_mu, d_sigma_theta);
    }

    protected double varInference4User(_User4ETBIR u){
        double current = 0.0, last = 1.0, converge = 0.0;
        int iter = 0;

        do{
            update_SigmaP(u);
            update_nu(u);

            current = calc_log_likelihood_per_user(u);
            if(iter > 0)
                converge = (last - current) / last;
            else
                converge = 1.0;

            last = current;
        } while(++iter < m_varMaxIter && Math.abs(converge) > m_varConverge);

        return current;
    }

    protected double varInference4Item(_Product4ETBIR i){
        double current = 0.0, last = 1.0, converge = 0.0;
        int iter = 0;

        double[] pNuStates = new double[number_of_topics];
        double[][] pSumStates = new double[number_of_topics][number_of_topics];
        Arrays.fill(pNuStates, 0.0);
        for(int k = 0; k < number_of_topics; k++)
            Arrays.fill(pSumStates[k], 0.0);

        ArrayList<Integer> Ui = m_mapByItem.get(m_itemsIndex.get(i.getID()));
        for (Integer userIdx : Ui) {
            _User4ETBIR user = (_User4ETBIR) m_users.get(userIdx);
            _Doc4ETBIR doc = (_Doc4ETBIR) m_corpus.getCollection().get(m_reviewIndex.get(
                    m_itemsIndex.get(i.getID()) + "_" + userIdx));
            for(int k = 0; k < number_of_topics; k++){
                for(int j = 0; j < number_of_topics; j++){
                    pNuStates[k] += user.m_nuP[j][k] * doc.m_mu[j];
                }

                for(int l = 0; l < number_of_topics; l++){
                    for (int j = 0; j < number_of_topics; j++){
                        pSumStates[k][l] += user.m_SigmaP[j][l][k] + user.m_nuP[j][k] * user.m_nuP[j][l];
                    }
                }
            }
        }

        do{
            update_eta(i, pNuStates, pSumStates);

            current = calc_log_likelihood_per_item(i);
            if (iter > 0)
                converge = (last - current) / last;
            else
                converge = 1.0;

            last = current;

        } while (++iter < m_varMaxIter && Math.abs(converge) > m_varConverge);

        return current;
    }

    protected double varInference4Doc(_Doc4ETBIR d, _User4ETBIR u, _Product4ETBIR i) {
        double current = 0.0, last = 1.0, converge = 0.0;
        int iter = 0;

        do {
            update_phi(d);
            update_zeta(d);
            update_mu(d, u ,i);
            update_SigmaTheta(d);
            update_zeta(d);

            current = calc_log_likelihood_per_doc(d, u, i);
            if (iter > 0)
                converge = (last-current) / last;
            else
                converge = 1.0;

            last = current;

        } while (++iter < m_varMaxIter && Math.abs(converge) > m_varConverge);

        return current;
    }

    //variational inference for p(z|w,\phi) for each document
    void update_phi(_Doc4ETBIR d){
        double logSum;
        int wid;
        _SparseFeature[] fv = d.getSparse();

        Arrays.fill(d.m_phiStat, 0);
        for (int n = 0; n < fv.length; n++) {
            wid = fv[n].getIndex();
            for (int k = 0; k < number_of_topics; k++)
                d.m_phi[n][k] = topic_term_probabilty[k][wid] + d.m_mu[k];

            // normalize
            logSum = Utils.logSum(d.m_phi[n]);
            for (int k = 0; k < number_of_topics; k++) {
                d.m_phi[n][k] = Math.exp(d.m_phi[n][k] - logSum);
                d.m_phiStat[k] += fv[n].getValue() * d.m_phi[n][k];
            }
        }
    }

    //variational inference for p(\theta|\mu,\Sigma) for each document
    void update_zeta(_Doc4ETBIR d){
        //estimate zeta
        d.m_zeta = 0;
        for (int k = 0; k < number_of_topics; k++)
            d.m_zeta += Math.exp(d.m_mu[k] + 0.5 * d.m_Sigma[k]);
    }

    // alternative: line search / fixed-stepsize gradient descent
    void update_mu(_Doc4ETBIR doc, _User4ETBIR user, _Product4ETBIR item){
        double fValue = 1.0, lastFValue = 1.0, cvg = 1e-4, diff, iterMax = 30, iter = 0;
        double stepsize = 1e-2, muG; // gradient for mu
        int N = doc.getTotalDocLength();

        double moment, zeta_stat = 1.0 / doc.m_zeta, norm;
        double etaSum = Utils.sumOfArray(item.m_eta);

        do {
            //update gradient of mu
            lastFValue = fValue;
            fValue = 0.0;
            for (int k = 0; k < number_of_topics; k++) {
                moment = N * zeta_stat * Math.exp(doc.m_mu[k] + 0.5 * doc.m_Sigma[k]);
                norm = Utils.dotProduct(item.m_eta, user.m_nuP[k]) / etaSum;

                muG = -m_rho * (doc.m_mu[k] - norm)
                        + doc.m_phiStat[k] - moment;//-1 because LBFGS is minimization

                fValue += -0.5 * m_rho * (doc.m_mu[k] * doc.m_mu[k] - 2 * doc.m_mu[k] * norm)
                        + doc.m_mu[k] * doc.m_phiStat[k] - moment;

                doc.m_mu[k] += stepsize * muG;//fixed stepsize
            }

            diff = (lastFValue - fValue) / lastFValue;
        } while (iter++ < iterMax && Math.abs(diff) > cvg);
    }

    void update_SigmaTheta(_Doc4ETBIR d){
        double fValue = 1.0, lastFValue = 1.0, cvg = 1e-4, diff, iterMax = 20, iter = 0;
        double stepsize = 1e-3, moment, sigma;
        double[] SigmaG = new double[number_of_topics]; // gradient for Sigma
        int N = d.getTotalDocLength();

        for(int k=0; k < number_of_topics; k++)
            d.m_sigmaSqrt[k] = Math.sqrt(d.m_Sigma[k]);

        do {

            //update gradient of sigma
            lastFValue = fValue;
            fValue = 0.0;

            for (int k = 0; k < number_of_topics; k++) {
                sigma = d.m_sigmaSqrt[k] * d.m_sigmaSqrt[k];
                moment = Math.exp(d.m_mu[k] + 0.5 * sigma);
                SigmaG[k] = -m_rho * d.m_sigmaSqrt[k] - N * d.m_sigmaSqrt[k] * moment / d.m_zeta
                        + 1.0 / d.m_sigmaSqrt[k]; //-1 because LBFGS is minimization
                fValue += -0.5 * m_rho * sigma - N * moment / d.m_zeta + 0.5 * Math.log(sigma);

                d.m_sigmaSqrt[k] += stepsize * SigmaG[k];//fixed stepsize
            }

            diff = (lastFValue - fValue) / lastFValue;
        } while(iter++ < iterMax && Math.abs(diff) > cvg);

        for(int k=0; k < number_of_topics; k++)
            d.m_Sigma[k] = d.m_sigmaSqrt[k] * d.m_sigmaSqrt[k];
    }

    //variational inference for p(P|\nu,\Sigma) for each user
    void update_SigmaP(_User4ETBIR u){
        ArrayList<Integer> Iu = m_mapByUser.get(m_usersIndex.get(u.getUserID()));
        RealMatrix eta_stat_sigma = MatrixUtils.createRealIdentityMatrix(number_of_topics).scalarMultiply(m_sigma);

        for (Integer itemIdx : Iu) {
            _Product4ETBIR item = (_Product4ETBIR) m_items.get(itemIdx);

            RealMatrix eta_vec = MatrixUtils.createColumnRealMatrix(item.m_eta);
            double eta_0 = Utils.sumOfArray(item.m_eta);
            RealMatrix eta_stat_i = MatrixUtils.createRealDiagonalMatrix(item.m_eta).add(
                    eta_vec.multiply(eta_vec.transpose()));

            eta_stat_sigma = eta_stat_sigma.add(eta_stat_i.scalarMultiply(m_rho / (eta_0 * (eta_0 + 1.0))));
        }
        eta_stat_sigma = new LUDecomposition(eta_stat_sigma).getSolver().getInverse();
        for (int k = 0; k < number_of_topics; k++) {
            u.m_SigmaP[k] = eta_stat_sigma.getData();
        }
    }

    //variational inference for p(P|\nu,\Sigma) for each user
    void update_nu(_User4ETBIR u){
        ArrayList<Integer> Iu = m_mapByUser.get(m_usersIndex.get(u.getUserID()));
        RealMatrix eta_stat_sigma = MatrixUtils.createRealMatrix(u.m_SigmaP[0]);

        for (int k = 0; k < number_of_topics; k++) {
            RealMatrix eta_stat_nu = MatrixUtils.createColumnRealMatrix(new double[number_of_topics]);

            for (Integer itemIdx : Iu) {
                _Product4ETBIR item = (_Product4ETBIR) m_items.get(itemIdx);
                _Doc4ETBIR d = (_Doc4ETBIR) m_corpus.getCollection().get(m_reviewIndex.get(itemIdx + "_"
                        + m_usersIndex.get(u.getUserID())));

                RealMatrix eta_vec = MatrixUtils.createColumnRealMatrix(item.m_eta);
                double eta_0 = Utils.sumOfArray(item.m_eta);
                eta_stat_nu = eta_stat_nu.add(eta_vec.scalarMultiply(d.m_mu[k] / eta_0));
            }
            u.m_nuP[k] = eta_stat_sigma.multiply(eta_stat_nu).scalarMultiply(m_rho).getColumn(0);
        }
    }

    // update eta with non-negative constraint using fix step graident descent
    void update_eta(_Product4ETBIR i, double[] pNuStates, double[][] pSumStates){
        double fValue = 1.0, lastFValue, cvg = 1e-4, diff, iterMax = 20, iter = 0;
        double stepsize=1e-2;

        double[] etaG = new double[number_of_topics];
        double[] eta_log = new double[number_of_topics];
        double[] eta_temp = new double[number_of_topics];
        for(int k = 0; k < number_of_topics; k++){
            eta_log[k] = Math.log(i.m_eta[k]);
            eta_temp[k] = i.m_eta[k];
        }
        do{
            lastFValue = fValue;
            fValue = 0.0;

            double eta0 = Utils.sumOfArray(eta_temp);
            double lgGammaEta = Utils.lgamma(eta0);
            double diGammaEta = Utils.digamma(eta0);
            double triGammaEta = Utils.trigamma(eta0);
            for(int k = 0; k < number_of_topics; k++) {
                double gTerm2 = 0.0;
                double gTerm3 = 0.0;
                double gTerm4 = 0.0;
                double term3 = 0.0;
                for(int l = 0; l < number_of_topics; l++){
                    gTerm2 += pNuStates[l] * eta_temp[l];
                    gTerm3 += 2 * pSumStates[l][k] * eta_temp[l];
                    for(int p = 0; p < number_of_topics; p++)
                        gTerm4 += eta_temp[l] * eta_temp[p] * pSumStates[l][p];
                    gTerm4 += eta_temp[l] * pSumStates[l][l];
                    term3 += eta_temp[l] * pSumStates[l][k];
                }
                gTerm3 += pSumStates[k][k];
                term3 += pSumStates[k][k];

                etaG[k] = Utils.trigamma(eta_temp[k]) * eta_temp[k] * (m_alpha[k] - eta_temp[k])
                        - triGammaEta * eta_temp[k] * (Utils.sumOfArray(m_alpha) - eta0)
                        + m_rho * eta_temp[k] * pNuStates[k] / eta0
                        - m_rho * eta_temp[k] * gTerm2 / (eta0 * eta0)
                        - m_rho * eta_temp[k] * gTerm3 / (2 * eta0 * (eta0 + 1.0))
                        + m_rho * (2 * eta0 + 1.0) * eta_temp[k] * gTerm4 / (2 * eta0 * eta0
                        * (eta0 + 1.0) * (eta0 + 1.0));

                fValue += (m_alpha[k] - eta_temp[k]) * (Utils.digamma(eta_temp[k]) - diGammaEta)
                        + Utils.lgamma(eta_temp[k])
                        + m_rho * eta_temp[k] * pNuStates[k] / eta0 - m_rho * eta_temp[k] * term3 / (2 * eta0 * (eta0 + 1.0));
            }
            fValue -=  lgGammaEta;
            // fix stepsize
            for(int k = 0; k < number_of_topics; k++) {
                eta_log[k] = eta_log[k] + stepsize * etaG[k];
                eta_temp[k] = Math.exp(eta_log[k]);
            }

            diff = (lastFValue - fValue) / lastFValue;
        }while(iter++ < iterMax && Math.abs(diff) > cvg);

        for(int k=0;k<number_of_topics;k++){
            i.m_eta[k] = eta_temp[k];
        }
    }

    @Override
    public void calculate_M_step(int iter) {
        //maximum likelihood estimation of p(w|z,\beta)
        for(int i=0; i<number_of_topics; i++) {
            double sum = Utils.sumOfArray(word_topic_sstat[i]);
            for(int v=0; v<vocabulary_size; v++) //will be in the log scale!!
                topic_term_probabilty[i][v] = Math.log(word_topic_sstat[i][v]/sum);
        }

        if (iter%5!=4)//no need to estimate \alpha very often
            return;

        //we need to estimate p(\theta|\alpha) as well later on
        int itemSize = m_items.size(), i = 0;
        double alphaSum, diAlphaSum, z, c, c1, c2, diff, deltaAlpha;
        do {
            alphaSum = Utils.sumOfArray(m_alpha);
            diAlphaSum = Utils.digamma(alphaSum);
            z = itemSize * Utils.trigamma(alphaSum);

            c1 = 0; c2 = 0;
            for(int k=0; k<number_of_topics; k++) {
                m_alphaG[k] = itemSize * (diAlphaSum - Utils.digamma(m_alpha[k])) + m_alphaStat[k];
                m_alphaH[k] = -itemSize * Utils.trigamma(m_alpha[k]);

                c1 +=  m_alphaG[k] / m_alphaH[k];
                c2 += 1.0 / m_alphaH[k];
            }
            c = c1 / (1.0/z + c2);

            diff = 0;
            for(int k=0; k<number_of_topics; k++) {
                deltaAlpha = (m_alphaG[k]-c) / m_alphaH[k];
                m_alpha[k] -= deltaAlpha;
                diff += deltaAlpha * deltaAlpha;
            }
            diff /= number_of_topics;
        } while(++i<m_varMaxIter && diff>m_varConverge);
    }

    // calculate the likelihood of user-related terms (term2-term7)
    protected double calc_log_likelihood_per_user(_User4ETBIR u){
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
    protected double calc_log_likelihood_per_item(_Product4ETBIR i){
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

        return log_likelihood;
    }

    // calculate the likelihood of doc-related terms (term3-term8 + term4-term9 + term5)
    protected double calc_log_likelihood_per_doc(_Doc4ETBIR doc, _User4ETBIR currentU, _Product4ETBIR currentI) {

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
        part3 += -m_rho * (0.5 * term1 - term2 / eta0 + term3 / (2 * eta0 * (eta0 + 1.0))) + number_of_topics/2.0
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
        for(int k = 0; k < number_of_topics; k++) {
            for (int n = 0; n < fv.length; n++) {
                wid = fv[n].getIndex();
                v = fv[n].getValue();
                term1 += v * doc.m_phi[n][k] * doc.m_mu[k];
                term3 += v * doc.m_phi[n][k] * Math.log(doc.m_phi[n][k]);
                part5 += v * doc.m_phi[n][k] * topic_term_probabilty[k][wid];
            }
            term2 += Math.exp(doc.m_mu[k] + doc.m_Sigma[k]/2.0);
        }
        part4 += term1 - doc.getTotalDocLength() * ( term2 / doc.m_zeta - 1.0 + Math.log(doc.m_zeta)) - term3;
        log_likelihood += part4;
        log_likelihood += part5;

        return log_likelihood;
    }

    protected void initialize_probability(Collection<_Doc> docs) {

        System.out.println("Initializing documents...");
        for(_Doc doc : docs)
            ((_Doc4ETBIR) doc).setTopics4Variational(number_of_topics, d_alpha, d_mu, d_sigma_theta);


        System.out.println("Initializing users...");
        for(int u_idx:m_mapByUser.keySet()){
            _User4ETBIR user = (_User4ETBIR) m_users.get(u_idx);
            user.setTopics4Variational(number_of_topics, d_nu, d_sigma_P);
        }


        System.out.println("Initializing items...");
        for(int i_idx : m_mapByItem.keySet()) {
            _Product4ETBIR item = (_Product4ETBIR) m_items.get(i_idx);
            item.setTopics4Variational(number_of_topics, d_alpha + 1);
        }

        // initialize with all smoothing terms
        init();
        Arrays.fill(m_alpha, d_alpha);

        System.out.println("Initializing model...");
        // initialize topic-word allocation, p(w|z)
        for(_Doc doc:docs) {
            updateStats4Doc((_Doc4ETBIR) doc);
        }

        for(int u_idx:m_mapByUser.keySet()) {
            _User4ETBIR user = (_User4ETBIR) m_users.get(u_idx);
            updateStats4User(user);
        }

        for(int i_idx : m_mapByItem.keySet()){
            _Product4ETBIR item = (_Product4ETBIR) m_items.get(i_idx);
            updateStats4Item(item);
        }

        calculate_M_step(0);
    }

    public void analyzeCorpus(){
        m_bipartite = new BipartiteAnalyzer(m_corpus);
        m_bipartite.analyzeCorpus();
        m_users = m_bipartite.getUsers();
        m_items = m_bipartite.getItems();
        m_usersIndex = m_bipartite.getUsersIndex();
        m_itemsIndex = m_bipartite.getItemsIndex();
        m_reviewIndex = m_bipartite.getReviewIndex();
    }

    @Override
    public void EMonCorpus() {
        m_trainSet = m_corpus.getCollection();
        //analyze corpus and generate bipartite
        analyzeCorpus();

        m_bipartite.analyzeBipartite(m_trainSet, "train");
        m_mapByUser = m_bipartite.getMapByUser();
        m_mapByItem = m_bipartite.getMapByItem();

        EM();
    }

    @Override
    public void EM(){
        System.out.format("%s\n", toString());
        initialize_probability(m_trainSet);

        int iter = 0;
        double lastAllLikelihood = 1.0;
        double currentAllLikelihood;
        double converge = 0.0;
        do{
            System.out.format("==========\nStart E step %d....\n", iter);
            if(m_multithread)
                currentAllLikelihood = multithread_E_step();
            else
                currentAllLikelihood = E_step();

            if(iter > 0)
                converge = (lastAllLikelihood - currentAllLikelihood) / lastAllLikelihood;
            else
                converge = 1.0;

            if(converge < 0){
//                m_varMaxIter += 10;
                System.out.println("! E_step not converge...");
            }
            if(Double.isNaN(currentAllLikelihood)){
                System.out.println("! E_step produces NaN likelihood...");
                break;
            }
            System.out.format("\n-------------\nStart M step %d....\n", iter);
            calculate_M_step(iter);
            lastAllLikelihood = currentAllLikelihood;
            System.out.format("[Stat]%s step: likelihood is %.3f, converge to %f...\n",
                    iter, currentAllLikelihood, converge);
            iter++;
            if(converge < m_converge)
                break;

        }while(iter < number_of_iteration && (converge < 0 || converge > m_converge));
    }


    //k-fold Cross Validation.
    public void crossValidation(int k) {
        analyzeCorpus();
        m_trainSet = new ArrayList<_Doc>();
        m_testSet = new ArrayList<_Doc>();

        double[] perf = new double[k];
        double[] like = new double[k];
        if(m_randomFold==true){
            m_corpus.shuffle(k);
            int[] masks = m_corpus.getMasks();
            ArrayList<_Doc> docs = m_corpus.getCollection();
            //Use this loop to iterate all the ten folders, set the train set and test set.
            for (int i = 0; i < k; i++) {
                for (int j = 0; j < masks.length; j++) {
                    if( masks[j]==i )
                        m_testSet.add(docs.get(j));
                    else
                        m_trainSet.add(docs.get(j));
                }

                System.out.println("Fold number "+i);
                System.out.println("Train Set Size "+m_trainSet.size());
                System.out.println("Test Set Size "+m_testSet.size());

                long start = System.currentTimeMillis();
                //train
                m_bipartite.analyzeBipartite(m_trainSet, "train");
                m_mapByUser = m_bipartite.getMapByUser();
                m_mapByItem = m_bipartite.getMapByItem();
                EM();

                //test
                m_bipartite.analyzeBipartite(m_testSet, "test");
                m_mapByUser_test = m_bipartite.getMapByUser_test();
                m_mapByItem_test = m_bipartite.getMapByItem_test();
                double[] results = EvaluatePerp();
                perf[i] = results[0];
                like[i] = results[1];

                System.out.format("%s Train/Test finished in %.2f seconds...\n", this.toString(), (System.currentTimeMillis()-start)/1000.0);
                m_trainSet.clear();
                m_testSet.clear();
            }
        }
        //output the performance statistics
        double mean = Utils.sumOfArray(perf)/k, var = 0;
        for(int i=0; i<perf.length; i++)
            var += (perf[i]-mean) * (perf[i]-mean);
        var = Math.sqrt(var/k);
        System.out.format("Perplexity %.3f+/-%.3f\n", mean, var);

        mean = Utils.sumOfArray(like)/k;
        var = 0;
        for(int i=0; i<like.length; i++)
            var += (like[i]-mean) * (like[i]-mean);
        var = Math.sqrt(var/k);
        System.out.format("Loglikelihood %.3f+/-%.3f\n", mean, var);
    }

    public int getTotalLength(){
        int length = 0;
        for(_Doc d:m_testSet){
            length += d.getTotalDocLength();
        }
        return length;
    }

    public double[] EvaluatePerp() {
        m_collectCorpusStats = false;
        double[] results = new double[2];
        double perplexity = 0, likelihood=0, log2 = Math.log(2.0), likelihood_doc = 0;
        double totalWords = 0.0;

        for(int u_idx : m_mapByUser_test.keySet()){
            m_mapByUser.get(u_idx).addAll(m_mapByUser_test.get(u_idx));
        }
        for(int i_idx : m_mapByItem_test.keySet()){
            m_mapByItem.get(i_idx).addAll(m_mapByItem_test.get(i_idx));
        }

        if (m_multithread) {
            System.out.println("In thread");
            likelihood_doc = multithread_inference();
            likelihood = likelihood_doc;
            perplexity = likelihood;
            totalWords = getTotalLength();

        } else {
            System.out.println("In Normal");
            int iter=0;
            double last = -1.0, converge = 0.0;
            do {
                init();
                likelihood = 0.0;
                for (_Doc d : m_testSet) {
                    likelihood += inference(d);
                }
                likelihood_doc = likelihood; //only count doc related likelihood for perplexity
                for (int u_idx : m_mapByUser_test.keySet()) {
                    _User4ETBIR user = (_User4ETBIR) m_users.get(u_idx);
                    likelihood += varInference4User(user);
                }
                for (int i_idx : m_mapByItem_test.keySet()) {
                    _Product4ETBIR item = (_Product4ETBIR) m_items.get(i_idx);
                    likelihood += varInference4Item(item);
                }
                if(iter > 0)
                    converge = Math.abs((likelihood - last) / last);
                else
                    converge = 1.0;

                last = likelihood;
                if(converge < m_varConverge)
                    break;
                System.out.print("---likelihood: " + last + "\n");
            }while(iter++<m_varMaxIter);
            likelihood = likelihood_doc;
            perplexity = likelihood;
            totalWords = getTotalLength();

        }
//		perplexity /= m_testSet.size();
        perplexity /= totalWords;
        perplexity = Math.exp(-perplexity);
        likelihood /= m_testSet.size();
        results[0] = perplexity;
        results[1] = likelihood;

        System.out.format("Test set perplexity is %.3f and log-likelihood is %.3f\n", perplexity, likelihood);

        return results;
    }

    @Override
    public void printParameterAggregation(int k, String folderName, String topicmodel){
        super.printParameterAggregation(k, folderName, topicmodel);
        printPara(folderName, "final", topicmodel);
        printTopWords(k, folderName + topicmodel + "_topWords.txt");
    }

    @Override
    public void printTopWords(int k, String topWordPath) {
        System.out.println("TopWord FilePath:" + topWordPath);
        Arrays.fill(m_sstat, 0);
        for(int d = 0; d < m_trainSet.size(); d++) {
            _Doc4ETBIR doc = (_Doc4ETBIR) m_trainSet.get(d);
            double expSum = 0;
            for(int i = 0; i < number_of_topics; i++){
                expSum += Math.exp(doc.m_mu[i]);
            }
            for(int i=0; i<number_of_topics; i++)
                m_sstat[i] += Math.exp(doc.m_mu[i]) / expSum;
        }
        Utils.L1Normalization(m_sstat);

        try{
            PrintWriter topWordWriter = new PrintWriter(new File(topWordPath));

            for(int i=0; i<topic_term_probabilty.length; i++) {
                MyPriorityQueue<_RankItem> fVector = new MyPriorityQueue<_RankItem>(k);
                for(int j = 0; j < vocabulary_size; j++)
                    fVector.add(new _RankItem(m_corpus.getFeature(j), topic_term_probabilty[i][j]));

                topWordWriter.format("Topic %d(%.5f):\t", i, m_sstat[i]);
                for(_RankItem it:fVector)
                    topWordWriter.format("%s(%.5f)\t", it.m_name, Math.exp(it.m_value));
                topWordWriter.write("\n");
            }
            topWordWriter.close();
        } catch(Exception ex){
            System.err.print("File Not Found");
        }
    }

    public void printPara(String folderName, String mode, String topicmodel){
        String etaFile = folderName + topicmodel + "_" + mode + "_eta4Item.txt";
        String pFile = folderName + topicmodel + "_" + mode + "_p4User.txt";
        try{
            PrintWriter etaWriter = new PrintWriter(new File(etaFile));

            for(int idx = 0; idx < m_items.size(); idx++) {
                etaWriter.write(String.format("item %d %s *********************\n", idx, m_items.get(idx).getID()));
                _Product4ETBIR item = (_Product4ETBIR) m_items.get(idx);
                etaWriter.format("-- eta: \n");
                for (int i = 0; i < number_of_topics; i++) {
                    etaWriter.format("%.8f\t", item.m_eta[i]);
                }
                etaWriter.write("\n");
            }
            etaWriter.close();
        } catch(Exception ex){
            System.err.print("File Not Found");
        }

        try{
            PrintWriter pWriter = new PrintWriter(new File(pFile));

            for(int idx = 0; idx < m_users.size(); idx++) {
                pWriter.write(String.format("user %d %s *********************\n", idx, m_users.get(idx).getUserID()));
                _User4ETBIR user = (_User4ETBIR) m_users.get(idx);
                for (int i = 0; i < number_of_topics; i++) {
                    pWriter.format("-- mu " + i + ": \n");
                    for(int j = 0; j < number_of_topics; j++) {
                        pWriter.format("%.5f\t", user.m_nuP[i][j]);
                    }
                    pWriter.write("\n");
                }
            }
            pWriter.close();
        } catch(Exception ex){
            System.err.print("File Not Found");
        }
    }

    @Override
    public HashMap<String, List<_Doc>> getDocByUser(){
        HashMap<String, List<_Doc>> docByUser = new HashMap<>();
        for(Integer uIdx : m_mapByUser.keySet()) {
            String userName = m_users.get(uIdx).getUserID();
            List<_Doc> docs = new ArrayList<>();
            for(Integer iIdx : m_mapByUser.get(uIdx)){
                docs.add(m_corpus.getCollection().get(m_reviewIndex.get(iIdx + "_" + uIdx)));
            }
            docByUser.put(userName, docs);
        }
        return docByUser;
    }

    @Override
    public HashMap<String, List<_Doc>> getDocByItem(){
        HashMap<String, List<_Doc>> docByItem = new HashMap<>();
        for(Integer iIdx : m_mapByItem.keySet()) {
            String itemName = m_items.get(iIdx).getID();
            List<_Doc> docs = new ArrayList<>();
            for(Integer uIdx : m_mapByItem.get(iIdx)){
                docs.add(m_corpus.getCollection().get(m_reviewIndex.get(iIdx + "_" + uIdx)));
            }
            docByItem.put(itemName, docs);
        }
        return docByItem;
    }

    @Override
    public void printTopWords(int k, String topWordPath, HashMap<String, List<_Doc>> docCluster) {
        try{
            PrintWriter topWordWriter = new PrintWriter(new File(topWordPath));
            PrintWriter topWordWriter2 = new PrintWriter(new File(topWordPath.replace(".txt","_est.txt")));

            for(Map.Entry<String, List<_Doc>> entryU : docCluster.entrySet()) {
                double[] gamma = new double[number_of_topics];
                double[] gamma_est = new double[number_of_topics];
                Arrays.fill(gamma, 0);
                Arrays.fill(gamma, 0);
                for(_Doc d:entryU.getValue()) {
                    _Doc4ETBIR doc = (_Doc4ETBIR) d;
                    double expSum = 0;
                    for(int i = 0; i < number_of_topics; i++){
                        expSum += Math.exp(doc.m_mu[i]);
                    }
                    for (int i = 0; i < number_of_topics; i++)
                        gamma[i] += Math.exp(doc.m_mu[i]) / expSum;

                    String userName = doc.getUserID();
                    String itemName = doc.getItemID();
                    _User4ETBIR user = (_User4ETBIR) m_users.get(m_usersIndex.get(userName));
                    _Product4ETBIR item = (_Product4ETBIR) m_items.get(m_itemsIndex.get(itemName));
                    double[] theta = new double[number_of_topics];
                    expSum = 0;
                    for(int i = 0; i < number_of_topics; i++){
                        theta[i] = Math.exp(Utils.dotProduct(user.m_nuP[i], item.m_eta));
                        expSum += theta[i];
                    }
                    for(int i = 0; i < number_of_topics; i++){
                        gamma_est[i] += theta[i] / expSum;
                    }

                }
                Utils.L1Normalization(gamma);
                Utils.L1Normalization(gamma_est);

                topWordWriter.format("ID %s(%d reviews)\n", entryU.getKey(), entryU.getValue().size());
                topWordWriter2.format("ID %s(%d reviews)\n", entryU.getKey(), entryU.getValue().size());
                for (int i = 0; i < topic_term_probabilty.length; i++) {
                    MyPriorityQueue<_RankItem> fVector = new MyPriorityQueue<_RankItem>(k);
                    for (int j = 0; j < vocabulary_size; j++)
                        fVector.add(new _RankItem(m_corpus.getFeature(j), topic_term_probabilty[i][j]));

                    topWordWriter.format("-- Topic %d(%.5f):\t", i, gamma[i]);
                    topWordWriter2.format("-- Topic %d(%.5f):\t", i, gamma_est[i]);
                    for (_RankItem it : fVector) {
                        topWordWriter.format("%s(%.5f)\t", it.m_name, m_logSpace ? Math.exp(it.m_value) : it.m_value);
                        topWordWriter2.format("%s(%.5f)\t", it.m_name, m_logSpace ? Math.exp(it.m_value) : it.m_value);
                    }
                    topWordWriter.write("\n");
                    topWordWriter2.write("\n");
                }
            }
            topWordWriter.close();
            topWordWriter2.close();
        } catch(Exception ex){
            System.err.println("File Not Found: " + topWordPath);
        }
    }

    public void loadPara(String filename){
        try {
            BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(filename), "UTF-8"));
            String line;

            String itemID = "";
            int idx = 0;
            while ((line = reader.readLine()) != null) {
                if(line.startsWith("ID")){
                    itemID = line.split("\\(")[0].split(" ")[1];
                    idx = 0;
                }else if(line.startsWith("-")){
                    String firststr = line.split("\\:")[0];
                    String secondstr = firststr.split("\\(")[1];
                    String thirdstr = secondstr.split("\\)")[0];
                    ((_Product4ETBIR)m_items.get(m_itemsIndex.get(itemID))).m_eta[idx++] = Double.parseDouble(
                            thirdstr);
                }
            }
            reader.close();
            System.out.format("Loading eta from %s\n", filename);
        } catch(IOException e){
            System.err.format("[Error]Failed to open file %s!!", filename);
        }
    }
}