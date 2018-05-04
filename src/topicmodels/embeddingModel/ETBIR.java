package topicmodels.embeddingModel;

import java.io.*;
import java.util.*;

import org.apache.commons.math3.linear.LUDecomposition;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;

import LBFGS.LBFGS;
import structures.MyPriorityQueue;
import structures._Corpus;
import structures._Doc;
import structures._Doc4ETBIR;
import structures._Product;
import structures._Product4ETBIR;
import structures._RankItem;
import structures._SparseFeature;
import structures._User;
import structures._User4ETBIR;
import topicmodels.LDA.LDA_Variational;
import utils.Utils;

/**
 * @author Lu Lin
 * Variational inference for Explainable Topic-Based Item Recommendation (ETBIR) model
 */
public class ETBIR extends LDA_Variational {

    /**
     * variables defined by base class
     */
//    protected int number_of_topics;
//    protected double m_converge;//relative change in log-likelihood to terminate EM
//    protected int vocabulary_size;
//    protected int number_of_iteration;//number of iterations in inferencing testing document
//    protected _Corpus m_corpus;
//    protected double d_beta;
//    protected double d_alpha; // smoothing of p(z|d)
//    protected double[][] topic_term_probabilty ; /* p(w|z) */
//
//    protected int m_varMaxIter;
//    protected double m_varConverge;
//	
//    protected double[] m_alpha; // we can estimate a vector of alphas as in p(\theta|\alpha)
//    protected double[] m_alphaStat; // statistics for alpha estimation
//    protected double[] m_alphaG; // gradient for alpha
//    protected double[] m_alphaH; // Hessian for alpha	

    protected int number_of_users;
    protected int number_of_items;

    protected List<_User4ETBIR> m_users;
    protected List<_Product4ETBIR> m_items;

    protected HashMap<String, Integer> m_usersIndex; //(userID, index in m_users)
    protected HashMap<String, Integer> m_itemsIndex; //(itemID, index in m_items)

    protected HashMap<Integer, ArrayList<Integer>>  m_mapByUser; //adjacent list for user
    protected HashMap<Integer, ArrayList<Integer>> m_mapByItem;
    protected HashMap<String, Integer> m_reviewIndex; //(itemIndex_userIndex, index in m_corpus.m_collection)

    protected double m_rho;
    protected double m_sigma;

    double m_pStats;
    double m_thetaStats;
    double m_eta_p_Stats;
    double m_eta_mean_Stats;

    double[] m_pNuStates;
    double[][] m_pSumStates;

    
    double d_mu = 1.0, d_sigma_theta = 0.1;
    double d_nu = 1.0, d_sigma_P = 0.1;
    
    public ETBIR(int emMaxIter, double emConverge,
                 double beta, _Corpus corpus, double lambda,
                 int number_of_topics, double alpha, int varMaxIter, double varConverge, //LDA_variational
                 double sigma, double rho) {
        super(emMaxIter, emConverge,
        	beta, corpus, lambda, 
        	number_of_topics, alpha, varMaxIter, varConverge);

        this.m_sigma = sigma;
        this.m_rho = rho;

        this.m_pNuStates = new double[this.number_of_topics];
        this.m_pSumStates = new double[this.number_of_topics][this.number_of_topics];
    }

    @Override
    public void EMonCorpus() {
        m_trainSet = m_corpus.getCollection();
        analyzeCorpus();
        EM();
    }

    public void analyzeCorpus(){
        System.out.print("Analzying review data in corpus");

        m_users = new ArrayList<>();
        m_items = new ArrayList<>();
        m_usersIndex = new HashMap<String, Integer>();
        m_itemsIndex = new HashMap<String, Integer>();
        m_reviewIndex = new HashMap<String, Integer>();
        m_mapByUser = new HashMap<Integer, ArrayList<Integer>>();
        m_mapByItem = new HashMap<Integer, ArrayList<Integer>>();

        int u_index = 0, i_index = 0, size = m_corpus.getCollection().size();
        for(int d = 0; d < size; d++){
            _Doc doc = m_corpus.getCollection().get(d);
            String userID = doc.getTitle();
            String itemID = doc.getItemID();

            if(!m_usersIndex.containsKey(userID)){
                m_users.add(new _User4ETBIR(userID));
                m_usersIndex.put(userID, u_index);
                m_mapByUser.put(u_index, new ArrayList<Integer>());
                u_index++;
            }

            if(!m_itemsIndex.containsKey(itemID)){
                m_items.add(new _Product4ETBIR(itemID));
                m_itemsIndex.put(itemID, i_index);
                m_mapByItem.put(i_index, new ArrayList<Integer>());
                i_index++;
            }

            int uIdx = m_usersIndex.get(userID);
            int iIdx = m_itemsIndex.get(itemID);
            m_mapByUser.get(uIdx).add(iIdx);
            m_mapByItem.get(iIdx).add(uIdx);

            m_reviewIndex.put(iIdx + "_" + uIdx, d);
            if ( (100 * d/size) % 10 == 0 )
            	System.out.print(".");//every 10%
        }
        System.out.println("Done!");//finish

        this.number_of_items = m_mapByItem.size();
        this.number_of_users = m_mapByUser.size();
        this.vocabulary_size = m_corpus.getFeatureSize();

        System.out.format("-- vocabulary size: %d\n -- corpus size: %d\n -- item size: %d\n-- user number: %d\n", 
        		vocabulary_size, size, number_of_items, number_of_users);
    }

    @Override
	protected void init() { // clear up for next iteration during EM
		super.init();		
        
        m_pStats = 0.0;
        m_thetaStats = 0.0;
        m_eta_p_Stats = 0.0;
        m_eta_mean_Stats = 0.0;

        Arrays.fill(m_pNuStates, 0.0);
        for(int k = 0; k < number_of_topics; k++)
            Arrays.fill(m_pSumStates[k], 0.0);

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
        _Product4ETBIR item = m_items.get(m_itemsIndex.get(doc.getItemID()));
        _User4ETBIR user = m_users.get(m_usersIndex.get(doc.getTitle()));
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
    
    // return log-likelihood
    @Override
    public double calculate_E_step(_Doc d){
    	_Doc4ETBIR doc = (_Doc4ETBIR)d;
    	
    	String userID = doc.getTitle();
        String itemID = doc.getItemID();
        _User4ETBIR currentU = m_users.get(m_usersIndex.get(userID));
        _Product4ETBIR currentI = m_items.get(m_itemsIndex.get(itemID));

        double cur = varInference4Doc(doc, currentU, currentI);        
        updateStats4Doc(doc);
        return cur;
    }

    protected double E_step(){
        int iter = 0;
        double totalLikelihood = 0.0, last = -1.0, converge = 0.0;

        do {
            infoWriter.write("-- E-step iter: " + iter + "\n");
            init();
            
            totalLikelihood = 0.0;
            int docNum = 0, userNum = 0, itemNum = 0;
            for (_Doc d:m_corpus.getCollection()) {
                infoWriter.write("---- doc: " + docNum + "\n");
                docNum++;
                totalLikelihood += calculate_E_step(d);
            }

            for (_User4ETBIR user:m_users) {
                infoWriter.write("---- user: " + userNum + "\n");
                userNum++;
            	totalLikelihood += varInference4User(user);
                updateStats4User(user);
            }

            for (_Product4ETBIR item : m_items) {
                infoWriter.write("---- item: " + itemNum + "\n");
                itemNum++;
            	totalLikelihood += varInference4Item(item);
                updateStats4Item(item);
            }

             if(iter > 0)
                converge = Math.abs((totalLikelihood - last) / last);
            else
                converge = 1.0;

            last = totalLikelihood;
            if(converge < m_varConverge)
                break;
            infoWriter.write(String.format("---- stats: m_pStat-%.5f, m_thetaStats-%.5f, " +
                            "m_eta_p_Stats-%.5f, m_eta_mean_Stats-%.5f \n",
                    m_pStats, m_thetaStats, m_eta_p_Stats, m_eta_mean_Stats) );
        }while(iter++ < m_varMaxIter);

        return totalLikelihood;
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
//            System.out.println("-- varInferencePerUser cur: " + current + "; converge: " + converge);

        } while(++iter < m_varMaxIter && Math.abs(converge) > m_varConverge);

        return current;
    }

    protected double varInference4Item(_Product4ETBIR i){
        double current = 0.0, last = 1.0, converge = 0.0;
        int iter = 0;

        Arrays.fill(m_pNuStates, 0.0);
        for(int k = 0; k < number_of_topics; k++)
            Arrays.fill(m_pSumStates[k], 0.0);

        ArrayList<Integer> Ui = m_mapByItem.get(m_itemsIndex.get(i.getID()));
        for (Integer userIdx : Ui) {
            _User4ETBIR user = m_users.get(userIdx);
            _Doc4ETBIR doc = (_Doc4ETBIR) m_corpus.getCollection().get(m_reviewIndex.get(
                    m_itemsIndex.get(i.getID()) + "_" + userIdx));
            for(int k = 0; k < number_of_topics; k++){
                for(int j = 0; j < number_of_topics; j++){
                    m_pNuStates[k] += user.m_nuP[j][k] * doc.m_mu[j];
                }

                for(int l = 0; l < number_of_topics; l++){
                    for (int j = 0; j < number_of_topics; j++){
                        m_pSumStates[k][l] += user.m_SigmaP[j][l][k] + user.m_nuP[j][k] * user.m_nuP[j][l];
                    }
                }
            }
        }

        do{
//            infoWriter.write("\t varIter: " + iter + "\n");
//            infoWriter.write("\t -- eta: " + Arrays.toString(i.m_eta) + "\n");
//            infoWriter.write("\t -- likelihood: " + String.valueOf(current) + "\n");

//            update_eta(i);

            current = calc_log_likelihood_per_item(i);
            if (iter > 0)
                converge = (last - current) / last;
            else
                converge = 1.0;

            last = current;
//            System.out.println("-- varInferencePerItem cur: " + current + "; converge: " + converge);

        } while (++iter < m_varMaxIter && Math.abs(converge) > m_varConverge);
//        infoWriter.write("\t varIter: " + iter + "\n");
//        infoWriter.write("\t -- eta: " + Arrays.toString(i.m_eta) + "\n");
//        infoWriter.write("\t -- likelihood: " + String.valueOf(current) + "\n");

        return current;
    }

    protected double varInference4Doc(_Doc4ETBIR d, _User4ETBIR u, _Product4ETBIR i) {
        double current = 0.0, last = 1.0, converge = 0.0;
        int iter = 0;

        do {
            infoWriter.write("\t varIter: " + iter + "\n");
            infoWriter.write("\t -- mu: " + Arrays.toString(d.m_mu) + "\n");
            infoWriter.write("\t -- sigma: " + Arrays.toString(d.m_Sigma) + "\n");
            infoWriter.write("\t -- likelihood: " + String.valueOf(current) + "\n");

            update_phi(d);
            update_zeta(d);
            update_mu(d, u ,i);
//            update_zeta(d);
            update_SigmaTheta(d);
            update_zeta(d);

            current = calc_log_likelihood_per_doc(d, u, i);
            if (iter > 0) 
                converge = (last-current) / last;
            else
                converge = 1.0;
            
            last = current;

//            System.out.println("-- varInferencePerDoc cur: " + current + "; converge: " + converge);
        } while (++iter < m_varMaxIter && Math.abs(converge) > m_varConverge);
        infoWriter.write("\t final varIter: " + iter + "\n");
        infoWriter.write("\t -- mu: " + Arrays.toString(d.m_mu) + "\n");
        infoWriter.write("\t -- sigma: " + Arrays.toString(d.m_Sigma) + "\n");
        infoWriter.write("\t -- likelihood: " + String.valueOf(current) + "\n");

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

            infoWriter.write("\t ---- sigmaSqrt: " + Arrays.toString(d.m_sigmaSqrt) + "\n");
            infoWriter.write("\t ---- sigmaThetaG: " + Arrays.toString(SigmaG) + "\n");

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
        infoWriter.write("\t ---- sigmaSqrt: " + Arrays.toString(d.m_sigmaSqrt) + "\n");
        infoWriter.write("\t ---- sigmaThetaG: " + Arrays.toString(SigmaG) + "\n");

        for(int k=0; k < number_of_topics; k++)
            d.m_Sigma[k] = d.m_sigmaSqrt[k] * d.m_sigmaSqrt[k];
    }

    //variational inference for p(P|\nu,\Sigma) for each user
    void update_SigmaP(_User4ETBIR u){
        ArrayList<Integer> Iu = m_mapByUser.get(m_usersIndex.get(u.getUserID()));
        RealMatrix eta_stat_sigma = MatrixUtils.createRealIdentityMatrix(number_of_topics).scalarMultiply(m_sigma);

        for (Integer itemIdx : Iu) {
            _Product4ETBIR item = m_items.get(itemIdx);

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

//        System.out.println("-- update nuP: origin: " + Arrays.toString(u.m_nuP[0]));
        for (int k = 0; k < number_of_topics; k++) {
            RealMatrix eta_stat_nu = MatrixUtils.createColumnRealMatrix(new double[number_of_topics]);

            for (Integer itemIdx : Iu) {
                _Product4ETBIR item = m_items.get(itemIdx);
                _Doc4ETBIR d = (_Doc4ETBIR) m_corpus.getCollection().get(m_reviewIndex.get(itemIdx + "_"
                        + m_usersIndex.get(u.getUserID())));

                RealMatrix eta_vec = MatrixUtils.createColumnRealMatrix(item.m_eta);
                double eta_0 = Utils.sumOfArray(item.m_eta);
                eta_stat_nu = eta_stat_nu.add(eta_vec.scalarMultiply(d.m_mu[k] / eta_0));
            }
            u.m_nuP[k] = eta_stat_sigma.multiply(eta_stat_nu).scalarMultiply(m_rho).getColumn(0);
        }
//        System.out.println("-- update nuP: origin: " + Arrays.toString(u.m_nuP[0]));
    }

    void update_eta_no_constraint(_Product4ETBIR i){
        ArrayList<Integer> Ui = m_mapByItem.get(m_itemsIndex.get(i.getID()));

        double fValue = 1.0, lastFValue = 1.0, cvg = 1e-6, diff, iterMax = 20, iter = 0;
        double stepsize = 1e3;
        double[] etaG = new double[number_of_topics];

        do{
            lastFValue = fValue;
            fValue = 0.0;
            double check;

            double eta0 = Utils.sumOfArray(i.m_eta);
            double lgGammaEta = Utils.lgamma(eta0);
            double diGammaEta = Utils.digamma(eta0);
            double triGammaEta = Utils.trigamma(eta0);
            for(int k = 0; k < number_of_topics; k++) {
                //might be optimized using global stats
                double gTerm1 = 0.0;
                double gTerm2 = 0.0;
                double gTerm3 = 0.0;
                double gTerm4 = 0.0;
                double term1 = 0.0;
                double term2 = 0.0;
                for (Integer userIdx : Ui) {
                    _User4ETBIR user = m_users.get(userIdx);
                    _Doc4ETBIR d = (_Doc4ETBIR) m_corpus.getCollection().get(m_reviewIndex.get(
                            m_itemsIndex.get(i.getID()) + "_" + userIdx));

                    for (int j = 0; j < number_of_topics; j++) {
                        gTerm1 += user.m_nuP[j][k] * d.m_mu[j];
                        term1 += i.m_eta[k] * user.m_nuP[j][k] * d.m_mu[j];

                        for (int l = 0; l < number_of_topics; l++) {
                            gTerm2 += i.m_eta[l] * user.m_nuP[j][l] * d.m_mu[j];

                            gTerm3 += i.m_eta[l] * (user.m_SigmaP[j][l][k]
                                    + user.m_nuP[j][l] * user.m_nuP[j][k]);
                            if (l == k) {
                                gTerm3 += (i.m_eta[l] + 1.0) * (user.m_SigmaP[j][l][k]
                                        + user.m_nuP[j][l] * user.m_nuP[j][k]);
                            }

                            term2 += i.m_eta[l] * i.m_eta[k] * (user.m_SigmaP[j][l][k]
                                    + user.m_nuP[j][l] * user.m_nuP[j][k]);
                            if (l == k) {
                                term2 += i.m_eta[k] * (user.m_SigmaP[j][k][k]
                                        + user.m_nuP[j][k] * user.m_nuP[j][k]);
                            }

                            for (int p = 0; p < number_of_topics; p++) {
                                gTerm4 += i.m_eta[l] * i.m_eta[p] * (user.m_SigmaP[j][l][p]
                                        + user.m_nuP[j][l] * user.m_nuP[j][p]);
                                if (p == l) {
                                    gTerm4 += i.m_eta[p] * (user.m_SigmaP[j][l][p]
                                            + user.m_nuP[j][l] * user.m_nuP[j][p]);
                                }
                            }
                        }
                    }
                }

                etaG[k] = -(Utils.trigamma(i.m_eta[k]) * (m_alpha[k] - i.m_eta[k])
                        - triGammaEta * (Utils.sumOfArray(m_alpha) - eta0)
                        + m_rho * gTerm1 / eta0 - m_rho * gTerm2 / (eta0 * eta0)
                        - m_rho * gTerm3 / (2 * eta0 * (eta0 + 1.0))
                        + m_rho * (2 * eta0 + 1.0) * gTerm4 / (2 * eta0 * eta0
                        * (eta0 + 1.0) * (eta0 + 1.0)));

                fValue += -((m_alpha[k] - i.m_eta[k]) * (Utils.digamma(i.m_eta[k]) - diGammaEta)
                        - lgGammaEta + Utils.lgamma(i.m_eta[k])
                        + m_rho * term1 / eta0 - m_rho * term2 / (2 * eta0 * (eta0 + 1.0)));

                if(k == 0) {
                    double eps = 1e-12;
                    i.m_eta[k] = i.m_eta[k] + eps;
                    double post = -((m_alpha[k] - i.m_eta[k]) * (Utils.digamma(i.m_eta[k]) - diGammaEta)
                            - lgGammaEta + Utils.lgamma(i.m_eta[k])
                            + m_rho * term1 / eta0 - m_rho * term2 / (2 * eta0 * (eta0 + 1.0)));

                    i.m_eta[k] = i.m_eta[k] - eps;
                    double pre = -((m_alpha[k] - i.m_eta[k]) * (Utils.digamma(i.m_eta[k]) - diGammaEta)
                            - lgGammaEta + Utils.lgamma(i.m_eta[k])
                            + m_rho * term1 / eta0 - m_rho * term2 / (2 * eta0 * (eta0 + 1.0)));
                    check = (post - pre) / eps;//gradient?
                }
            }
            // fix stepsize
            for(int k = 0; k < number_of_topics; k++)
                i.m_eta[k] = i.m_eta[k] - stepsize * etaG[k];

            diff = (lastFValue - fValue) / lastFValue;
        }while(iter++ < iterMax && Math.abs(diff) > cvg);
    }

    // update eta with non-negative constraint using fix step graident descent
    void update_eta(_Product4ETBIR i){

        double fValue = 1.0, lastFValue, cvg = 1e-4, diff, iterMax = 20, iter = 0;
        double stepsize=1e-3;

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
                    gTerm2 += m_pNuStates[l] * eta_temp[l];
                    gTerm3 += 2 * m_pSumStates[l][k] * eta_temp[l];
                    for(int p = 0; p < number_of_topics; p++)
                        gTerm4 += eta_temp[l] * eta_temp[p] * m_pSumStates[l][p];
                    gTerm4 += eta_temp[l] * m_pSumStates[l][l];
                    term3 += eta_temp[l] * m_pSumStates[l][k];
                }
                gTerm3 += m_pSumStates[k][k];
                term3 += m_pSumStates[k][k];

                etaG[k] = Utils.trigamma(eta_temp[k]) * eta_temp[k] * (m_alpha[k] - eta_temp[k])
                        - triGammaEta * eta_temp[k] * (Utils.sumOfArray(m_alpha) - eta0)
                        + m_rho * eta_temp[k] * m_pNuStates[k] / eta0
                        - m_rho * eta_temp[k] * gTerm2 / (eta0 * eta0)
                        - m_rho * eta_temp[k] * gTerm3 / (2 * eta0 * (eta0 + 1.0))
                        + m_rho * (2 * eta0 + 1.0) * eta_temp[k] * gTerm4 / (2 * eta0 * eta0
                        * (eta0 + 1.0) * (eta0 + 1.0));

                fValue += (m_alpha[k] - eta_temp[k]) * (Utils.digamma(eta_temp[k]) - diGammaEta)
                        + Utils.lgamma(eta_temp[k])
                        + m_rho * eta_temp[k] * m_pNuStates[k] / eta0 - m_rho * eta_temp[k] * term3 / (2 * eta0 * (eta0 + 1.0));
            }
            fValue -=  lgGammaEta;
            // fix stepsize
            for(int k = 0; k < number_of_topics; k++) {
                eta_log[k] = eta_log[k] + stepsize * etaG[k];
                eta_temp[k] = Math.exp(eta_log[k]);
            }

            diff = (lastFValue - fValue) / lastFValue;
        }while(iter++ < iterMax && Math.abs(diff) > cvg);

        infoWriter.write("\t -- etaG: " + Arrays.toString(etaG) + "\n");
        for(int k=0;k<number_of_topics;k++){
            i.m_eta[k] = eta_temp[k];
        }
    }

    // update eta with non-negative constraint using lbfgs
    void update_eta_lbfgs(_Product4ETBIR i){
        int[] iflag = {0}, iprint = {-1,3};
        ArrayList<Integer> Ui = m_mapByItem.get(m_itemsIndex.get(i.getID()));

        double fValue, iterMax = 20, iter = 0;

        double[] etaG = new double[number_of_topics];
        double[] eta_log = new double[number_of_topics];
        double[] eta_temp = new double[number_of_topics];
        double[] eta_diag = new double[number_of_topics];
        for(int k = 0; k < number_of_topics; k++){
            eta_log[k] = Math.log(i.m_eta[k]);
            eta_diag[k] = eta_log[k];
            eta_temp[k] = i.m_eta[k];
        }

        try {
            do {
                fValue = 0.0;
                for(int k = 0; k < number_of_topics; k++){
                    eta_temp[k] = Math.exp(eta_log[k]);
                }

                double eta0 = Utils.sumOfArray(eta_temp);
                double lgGammaEta = Utils.lgamma(eta0);
                double triGammaEta = Utils.trigamma(eta0);
                double diGammaEta = Utils.digamma(eta0);
                for (int k = 0; k < number_of_topics; k++) {
                    double gTerm1 = 0.0;
                    double gTerm2 = 0.0;
                    double gTerm3 = 0.0;
                    double gTerm4 = 0.0;
                    double term1 = 0.0;
                    double term2 = 0.0;
                    for (Integer userIdx : Ui) {
                        _User4ETBIR user = m_users.get(userIdx);
                        _Doc4ETBIR d = (_Doc4ETBIR) m_corpus.getCollection().get(m_reviewIndex.get(
                                m_itemsIndex.get(i.getID()) + "_" + userIdx));

                        for (int j = 0; j < number_of_topics; j++) {
                            gTerm1 += user.m_nuP[j][k] * d.m_mu[j];
                            term1 += eta_temp[k] * user.m_nuP[j][k] * d.m_mu[j];

                            for (int l = 0; l < number_of_topics; l++) {
                                gTerm2 += eta_temp[l] * user.m_nuP[j][l] * d.m_mu[j];

                                gTerm3 += eta_temp[l] * (user.m_SigmaP[j][l][k] + user.m_SigmaP[j][k][l]
                                        + 2 * user.m_nuP[j][l] * user.m_nuP[j][k]);
                                if (l == k) {
                                    gTerm3 += (user.m_SigmaP[j][l][k]
                                            + user.m_nuP[j][l] * user.m_nuP[j][k]);
                                }

                                term2 += eta_temp[l] * eta_temp[k] * (user.m_SigmaP[j][l][k]
                                        + user.m_nuP[j][l] * user.m_nuP[j][k]);
                                if (l == k) {
                                    term2 += eta_temp[k] * (user.m_SigmaP[j][k][k]
                                            + user.m_nuP[j][k] * user.m_nuP[j][k]);
                                }

                                for (int p = 0; p < number_of_topics; p++) {
                                    gTerm4 += eta_temp[l] * eta_temp[p] * (user.m_SigmaP[j][l][p]
                                            + user.m_nuP[j][l] * user.m_nuP[j][p]);
                                    if (p == l) {
                                        gTerm4 += eta_temp[p] * (user.m_SigmaP[j][l][p]
                                                + user.m_nuP[j][l] * user.m_nuP[j][p]);
                                    }
                                }
                            }
                        }
                    }

                    etaG[k] = -(Utils.trigamma(eta_temp[k]) * eta_temp[k] * (m_alpha[k] - eta_temp[k])
                            - triGammaEta * eta_temp[k] * (Utils.sumOfArray(m_alpha) - eta0)
                            + m_rho * eta_temp[k] * gTerm1 / eta0
                            - m_rho * eta_temp[k] * gTerm2 / (eta0 * eta0)
                            - m_rho * eta_temp[k] * gTerm3 / (2 * eta0 * (eta0 + 1.0))
                            + m_rho * (2 * eta0 + 1.0) * eta_temp[k] * gTerm4 / (2 * eta0 * eta0
                            * (eta0 + 1.0) * (eta0 + 1.0)));

                    fValue += -((m_alpha[k] - eta_temp[k]) * (Utils.digamma(eta_temp[k]) - diGammaEta)
                            + Utils.lgamma(eta_temp[k])
                            + m_rho * term1 / eta0 - m_rho * term2 / (2 * eta0 * (eta0 + 1.0)));

                }
                fValue -= lgGammaEta;
                LBFGS.lbfgs(number_of_topics, 4, eta_log, fValue, etaG, false, eta_diag, iprint, 1e-6, 1e-32, iflag);

            } while (iter++ < iterMax && iflag[0] != 0);
        }catch(LBFGS.ExceptionWithIflag e){
            e.printStackTrace();
        }

        for(int k=0;k<number_of_topics;k++){
            i.m_eta[k] = Math.exp(eta_log[k]);
        }
    }

    @Override
	public void calculate_M_step(int iter) {
    	super.calculate_M_step(iter);
        infoWriter.write("-- M step: " + "\n");

        //maximize likelihood for \rho of p(\theta|P\gamma, \rho)
//        m_rho = number_of_topics / (m_thetaStats + m_eta_p_Stats - 2 * m_eta_mean_Stats);

        //maximize likelihood for \sigma
//        m_sigma = number_of_topics / m_pStats;
    }
    
    @Override
    protected int getCorpusSize() {
		return number_of_items;
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
        part3 += -m_rho * (0.5 * term1 - term2 / eta0 + term3 / (eta0 * (eta0 + 1.0))) + number_of_topics/2.0
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
                part5 += v * doc.m_phi[n][k] * topic_term_probabilty[k][wid];
            }
            term2 += Math.exp(doc.m_mu[k] + doc.m_Sigma[k]/2.0);
        }
        part4 += term1 - doc.getTotalDocLength() * ( term2 / doc.m_zeta - 1.0 + Math.log(doc.m_zeta)) - term3;
        log_likelihood += part4;
        log_likelihood += part5;

        return log_likelihood;
    }

    //create space; initial parameters
    public void initModel(){

        //initialize parameters
        Random r = new Random();
        double val = 0.0;
        for(int k = 0; k < number_of_topics; k++){
            double sum = 0.0;
            for(int v = 0; v < vocabulary_size; v++){
                val = r.nextDouble() + d_beta;
                sum += val;
                topic_term_probabilty[k][v] = val;
            }

            for(int v = 0; v < vocabulary_size; v++){
                topic_term_probabilty[k][v] = Math.log(topic_term_probabilty[k][v]) - Math.log(sum);
            }
        }
    }

    protected void initialize_probability(List<_Doc> docs, List<_User4ETBIR> users, List<_Product4ETBIR> items) {
        // initialize with all smoothing terms
        init();

        // initialize topic-word allocation, p(w|z)
        for(_Doc d:docs) {
            ((_Doc4ETBIR) d).setTopics4Variational(number_of_topics, d_alpha, d_mu, d_sigma_theta);//allocate memory and randomize it
            updateStats4Doc((_Doc4ETBIR) d);
        }

        for(_User u:users) {
            ((_User4ETBIR) u).setTopics4Variational(number_of_topics, d_nu, d_sigma_P);
            updateStats4User((_User4ETBIR) u);
        }

        for(_Product i:items){
            ((_Product4ETBIR) i).setTopics4Variational(number_of_topics, d_alpha);
            updateStats4Item((_Product4ETBIR) i);
        }

        calculate_M_step(0);
    }

    @Override
    public void EM(){

        System.out.println("Initializing documents...");
        for(_Doc doc : m_corpus.getCollection())
            ((_Doc4ETBIR) doc).setTopics4Variational(number_of_topics, d_alpha, d_mu, d_sigma_theta);


        System.out.println("Initializing users...");
        for(_User user : m_users)
            ((_User4ETBIR) user).setTopics4Variational(number_of_topics, d_nu, d_sigma_P);


        System.out.println("Initializing items...");
        for(_Product item : m_items)
            ((_Product4ETBIR) item).setTopics4Variational(number_of_topics, d_alpha);

        System.out.println("Initializing model...");
        initialize_probability(m_corpus.getCollection(), m_users, m_items);

        loadPara("./myData/yelp/byUser_40_50_12/output/feature0_new/LDA_Variational_gammaByUser.txt");

        int iter = 0;
        double lastAllLikelihood = 1.0;
        double currentAllLikelihood;
        double converge = 0.0;
        do{
            infoWriter.write("EM iter: " + iter + "\n");
            currentAllLikelihood = E_step();

            if(iter > 0)
                converge = (lastAllLikelihood - currentAllLikelihood) / lastAllLikelihood;
            else
                converge = 1.0;

            if(converge < 0){
//                m_varMaxIter += 10;
                System.out.println("! E_step not converge...");
            }
            calculate_M_step(iter);
            lastAllLikelihood = currentAllLikelihood;
            System.out.format("%s step: likelihood is %.3f, converge to %f...\n",
                    iter, currentAllLikelihood, converge);
            iter++;
            if(converge < m_converge)
                break;

            System.out.println("sigma: " + m_sigma + "; rho: " + m_rho);
            System.out.println(Utils.sumOfArray(word_topic_sstat[0]));
        }while(iter < number_of_iteration && (converge < 0 || converge > m_converge));
    }

    @Override
    public void printParameterAggregation(int k, String folderName, String topicmodel){
        super.printParameterAggregation(k, folderName, topicmodel);

        String etaFile = folderName + topicmodel + "_eta4Item.txt";
        String pFile = folderName + topicmodel + "_p4User.txt";
        try{
            PrintWriter etaWriter = new PrintWriter(new File(etaFile));

            for(int idx = 0; idx < m_items.size(); idx++) {
                etaWriter.write("item " + idx + "*************\n");
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
                pWriter.write("user " + idx + "*************\n");
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
    public void printTopWords(int k, String topWordPath) {
        System.out.println("TopWord FilePath:" + topWordPath);
        Arrays.fill(m_sstat, 0);
        for(int d = 0; d < m_corpus.getCollection().size(); d++) {
            _Doc4ETBIR doc = (_Doc4ETBIR) m_corpus.getCollection().get(d);
            for(int i=0; i<number_of_topics; i++)
                m_sstat[i] += Math.exp(doc.m_mu[i]);
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

    public void loadPara(String filename){
        try {
            BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(filename), "UTF-8"));
            String line;

            String itemID = "";
            int idx = 0;
            while ((line = reader.readLine()) != null) {
                if(line.startsWith("ID")){
                    itemID = line.split("\t")[1];
                    idx = 0;
                }else if(line.startsWith("-")){
                    m_items.get(m_itemsIndex.get(itemID)).m_eta[idx++] = Double.parseDouble(
                            line.split(":")[0].split("\\(")[1].split("\\)")[0]);
                }
            }
            reader.close();
            System.out.format("Loading eta from %s\n", filename);
        } catch(IOException e){
            System.err.format("[Error]Failed to open file %s!!", filename);
        }
    }

}