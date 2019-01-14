package topicmodels.LDA;

import Analyzer.BipartiteAnalyzer;
import structures.*;
import topicmodels.LDA.LDA_Variational;
import topicmodels.markovmodel.HTSM;
import topicmodels.multithreads.LDA.LDA_Focus_multithread;
import topicmodels.multithreads.TopicModelWorker;
import utils.Utils;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.*;

public class LDA_Focus extends LDA_Variational {
    protected List<_User> m_users;
    protected List<_Product4ETBIR> m_items;

    protected HashMap<String, Integer> m_usersIndex; //(userID, index in m_users)
    protected HashMap<String, Integer> m_itemsIndex; //(itemID, index in m_items)
    protected HashMap<String, Integer> m_reviewIndex; //(itemIndex_userIndex, index in m_corpus.m_collection)

    //training set of users and items
    protected HashMap<Integer, ArrayList<Integer>>  m_mapByUser; //adjacent list for user, controlled by m_testFlag.
    protected HashMap<Integer, ArrayList<Integer>> m_mapByItem;

    //testing set of users and items
    protected HashMap<Integer, ArrayList<Integer>> m_mapByUser_test; //test
    protected HashMap<Integer, ArrayList<Integer>> m_mapByItem_test;

    protected BipartiteAnalyzer m_bipartite;

    protected String m_mode;

    protected double[][] m_alphaList; // we can estimate a vector of alphas as in p(\theta|\alpha)
    protected double[][] m_alphaStatList; // statistics for alpha estimation

    //coldstart_all, coldstart_user, coldstart_item, normal;
    protected double[] m_likelihood_array;
    protected double[] m_perplexity_array;
    protected double[] m_totalWords_array;
    protected double[] m_docSize_array;

    public LDA_Focus(int number_of_iteration, double converge,
                           double beta, _Corpus c, double lambda,
                           int number_of_topics, double alpha, int varMaxIter, double varConverge) {
        super(number_of_iteration, converge, beta, c, lambda, number_of_topics, alpha, varMaxIter, varConverge);
    }

    public void setMode(String mode){
        this.m_mode = mode;
    }

    @Override
    public String toString() {
        return String.format("LDA_%s[k:%d, alpha:%.2f, beta:%.2f, Variational]", m_mode, number_of_topics, d_alpha, d_beta);
    }

    public void analyzeCorpus(){
        m_bipartite = new BipartiteAnalyzer(m_corpus);
        m_bipartite.analyzeCorpus();
        m_users = m_bipartite.getUsers();
        m_items = m_bipartite.getItems();
        m_usersIndex = m_bipartite.getUsersIndex();
        m_itemsIndex = m_bipartite.getItemsIndex();
        m_reviewIndex = m_bipartite.getReviewIndex();
        //allocate space for focused alpha
        if(m_mode.equals("User")) {
            m_alphaList = new double[m_users.size()][number_of_topics];
            m_alphaStatList = new double[m_users.size()][number_of_topics];
        }else if(m_mode.equals("Item")){
            m_alphaStatList = new double[m_items.size()][number_of_topics];
            m_alphaList = new double[m_items.size()][number_of_topics];
        }
    }

    @Override
    protected void initialize_probability(Collection<_Doc> collection) {
        // initialize with all smoothing terms
        init();
        for(int i = 0; i < m_alphaList.length;i++) {
            Arrays.fill(m_alphaList[i], d_alpha);
        }

        // initialize topic-word allocation, p(w|z)
        for(_Doc d:collection) {
            d.setTopics4Variational(number_of_topics, d_alpha);//allocate memory and randomize it
            collectStats(d);
        }

        calculate_M_step(0);
    }

    protected int getAlphaIdx(_Doc d){
        int idx=0;
        if(m_mode.equals("User"))
            idx = m_usersIndex.get(((_Review)d).getUserID());
        else if(m_mode.equals("Item"))
            idx = m_itemsIndex.get(((_Review)d).getItemID());
        return idx;
    }

    @Override
    protected void init() {//will be called at the beginning of each EM iteration
        // initialize alpha statistics
        for(int i = 0; i < m_alphaStatList.length;i++) {
            Arrays.fill(m_alphaStatList[i], 0);
        }

        // initialize with all smoothing terms
        for(int i=0; i<number_of_topics; i++)
            Arrays.fill(word_topic_sstat[i], d_beta-1.0);
        imposePrior();
    }

    protected void collectStats(_Doc d) {
        _SparseFeature[] fv = d.getSparse();
        int wid;
        double v;
        for(int n=0; n<fv.length; n++) {
            wid = fv[n].getIndex();
            v = fv[n].getValue();
            for(int i=0; i<number_of_topics; i++)
                word_topic_sstat[i][wid] += v*d.m_phi[n][i];
        }

        //if we need to use maximum likelihood to estimate alpha
        double diGammaSum = Utils.digamma(Utils.sumOfArray(d.m_sstat));
        for(int i=0; i<number_of_topics; i++)
            m_alphaStatList[getAlphaIdx(d)][i] += Utils.digamma(d.m_sstat[i]) - diGammaSum;
    }

    @Override
    public double calculate_E_step(_Doc d) {
        double last = 1;
        if (m_varConverge>0)
            last = calculate_log_likelihood(d);

        double current = last, converge, logSum, v;
        int iter = 0, wid;
        _SparseFeature[] fv = d.getSparse();

        do {
            //variational inference for p(z|w,\phi)
            for(int n=0; n<fv.length; n++) {
                wid = fv[n].getIndex();
                v = fv[n].getValue();
                for(int i=0; i<number_of_topics; i++)
                    d.m_phi[n][i] = topic_term_probabilty[i][wid] + Utils.digamma(d.m_sstat[i]);

                logSum = Utils.logSum(d.m_phi[n]);
                for(int i=0; i<number_of_topics; i++)
                    d.m_phi[n][i] = Math.exp(d.m_phi[n][i] - logSum);
            }

            //variational inference for p(\theta|\gamma)
            System.arraycopy(m_alphaList[getAlphaIdx(d)], 0, d.m_sstat, 0, m_alphaList[getAlphaIdx(d)].length);
            for(int n=0; n<fv.length; n++) {
                v = fv[n].getValue();
                for(int i=0; i<number_of_topics; i++)
                    d.m_sstat[i] += d.m_phi[n][i] * v;
            }

            if (m_varConverge>0) {
                current = calculate_log_likelihood(d);
                converge = Math.abs((current - last)/last);
                last = current;

                if (converge<m_varConverge)
                    break;
            }
        } while(++iter<m_varMaxIter);

        if (m_collectCorpusStats) {
            collectStats(d);//collect the sufficient statistics after convergence
            return current;
        } else if (m_varConverge>0)
            return current;//to avoid computing this again
        else
            return calculate_log_likelihood(d);//in testing, we need to compute log-likelihood
    }

    @Override
    public void calculate_M_step(int iter) {
        //maximum likelihood estimation of p(w|z,\beta)
        for(int i=0; i<number_of_topics; i++) {
            double sum = Utils.sumOfArray(word_topic_sstat[i]);
            for(int v=0; v<vocabulary_size; v++) //will be in the log scale!!
                topic_term_probabilty[i][v] = Math.log(word_topic_sstat[i][v]/sum);
        }

        //we need to estimate p(\theta|\alpha) as well later on
        for(Integer idx : m_mode.equals("User") ? m_mapByUser.keySet() : m_mapByItem.keySet()) {
            int docSize = m_mode.equals("User") ? m_mapByUser.get(idx).size() : m_mapByItem.get(idx).size();
            int i = 0;
            double alphaSum, diAlphaSum, z, c, c1, c2, diff, deltaAlpha;
            do {
                alphaSum = Utils.sumOfArray(m_alphaList[idx]);
                diAlphaSum = Utils.digamma(alphaSum);
                z = docSize * Utils.trigamma(alphaSum);

                c1 = 0;
                c2 = 0;
                for (int k = 0; k < number_of_topics; k++) {
                    m_alphaG[k] = docSize * (diAlphaSum - Utils.digamma(m_alphaList[idx][k])) + m_alphaStatList[idx][k];
                    m_alphaH[k] = -docSize * Utils.trigamma(m_alphaList[idx][k]);

                    c1 += m_alphaG[k] / m_alphaH[k];
                    c2 += 1.0 / m_alphaH[k];
                }
                c = c1 / (1.0 / z + c2);

                diff = 0;
                for (int k = 0; k < number_of_topics; k++) {
                    deltaAlpha = (m_alphaG[k] - c) / m_alphaH[k];
                    m_alphaList[idx][k] -= 0.001 * deltaAlpha; // set small stepsize, so the value won't jump too much
                    diff += deltaAlpha * deltaAlpha;
                }
                diff /= number_of_topics;
            } while (++i < m_varMaxIter && diff > m_varConverge);
        }

        // update per-document topic distribution vectors
        finalEst();
    }

    @Override
    public double calculate_log_likelihood(_Doc d) {
        int wid;
        double[] diGamma = new double[this.number_of_topics];
        double logLikelihood = Utils.lgamma(Utils.sumOfArray(m_alphaList[getAlphaIdx(d)])) - Utils.lgamma(Utils.sumOfArray(d.m_sstat)), v;
        double diGammaSum = Utils.digamma(Utils.sumOfArray(d.m_sstat));
        for(int i=0; i<number_of_topics; i++) {
            diGamma[i] = Utils.digamma(d.m_sstat[i]) - diGammaSum;
            logLikelihood += Utils.lgamma(d.m_sstat[i]) - Utils.lgamma(m_alphaList[getAlphaIdx(d)][i])
                    + (m_alphaList[getAlphaIdx(d)][i] - d.m_sstat[i]) * diGamma[i];
        }

        //collect the sufficient statistics
        _SparseFeature[] fv = d.getSparse();
        for(int n=0; n<fv.length; n++) {
            wid = fv[n].getIndex();
            v = fv[n].getValue();
            for(int i=0; i<number_of_topics; i++)
                logLikelihood += v * d.m_phi[n][i] * (diGamma[i] + topic_term_probabilty[i][wid] - Math.log(d.m_phi[n][i]));
        }

        return logLikelihood;
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
    public double[] oneFoldValidation(){
        analyzeCorpus();
        m_trainSet = new ArrayList<_Doc>();
        m_testSet = new ArrayList<_Doc>();
        for(_Doc d:m_corpus.getCollection()){
            if(d.getType() == _Doc.rType.TRAIN){
                m_trainSet.add(d);
            }else if(d.getType() == _Doc.rType.TEST){
                m_testSet.add(d);
            }
        }

        System.out.format("train size = %d, test size = %d....\n", m_trainSet.size(), m_testSet.size());

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

        double[] results = Evaluation3();
        System.out.format("[Info]%s Train/Test finished in %.2f seconds...\n", this.toString(), (System.currentTimeMillis()-start)/1000.0);

        return results;
    }

    @Override
    public double[] Evaluation2() {
        analyzeCorpus();

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

        double[] results = Evaluation3();
        System.out.format("[Info]%s Train/Test finished in %.2f seconds...\n", this.toString(), (System.currentTimeMillis()-start)/1000.0);

        double[] result_all = new double[2];
        result_all[0] = results[8];
        result_all[1] = results[9];

        return result_all;
    }

    public double[] Evaluation3() {
        m_collectCorpusStats = false;
        m_likelihood_array = new double[5];
        m_perplexity_array = new double[5];
        m_totalWords_array = new double[5];
        m_docSize_array = new double[5];

        double loglikelihood, log2 = Math.log(2.0);

        if (m_multithread) {
            multithread_inference();
            System.out.println("[Info]Start evaluation in thread");
            for(TopicModelWorker worker:m_workers) {
                worker = (LDA_Focus_multithread.LDA_Focus_worker) worker;
                Utils.add2Array(m_likelihood_array, ((LDA_Focus_multithread.LDA_Focus_worker) worker).getLogLikelihoodArray(), 1);
                Utils.add2Array(m_perplexity_array, ((LDA_Focus_multithread.LDA_Focus_worker) worker).getPerplexityArray(), 1);
                Utils.add2Array(m_totalWords_array, ((LDA_Focus_multithread.LDA_Focus_worker) worker).getTotalWordsArray(), 1);
                Utils.add2Array(m_docSize_array, ((LDA_Focus_multithread.LDA_Focus_worker) worker).getDocSizeArray(), 1);
            }
        } else {
            System.out.println("[Info]Start evaluation in Normal");
            Arrays.fill(m_likelihood_array, 0);
            Arrays.fill(m_perplexity_array, 0);
            Arrays.fill(m_totalWords_array,0);
            Arrays.fill(m_docSize_array, 0);
            for(_Doc d:m_testSet) {
                loglikelihood = inference(d);

                if(!m_mapByUser.containsKey(m_usersIndex.get(((_Review)d).getUserID()))
                        && !m_mapByItem.containsKey(m_itemsIndex.get(((_Review)d).getItemID()))){//all coldstart
                    m_likelihood_array[0] += loglikelihood;
                    m_perplexity_array[0] += loglikelihood;
                    m_totalWords_array[0] += d.getTotalDocLength();
                    m_docSize_array[0] += 1;
                }else if(!m_mapByUser.containsKey(m_usersIndex.get(((_Review)d).getUserID()))
                        && m_mapByItem.containsKey(m_itemsIndex.get(((_Review)d).getItemID()))){//user coldstart
                    m_likelihood_array[1] += loglikelihood;
                    m_perplexity_array[1] += loglikelihood;
                    m_totalWords_array[1] += d.getTotalDocLength();
                    m_docSize_array[1] += 1;
                }else if(m_mapByUser.containsKey(m_usersIndex.get(((_Review)d).getUserID()))
                        && !m_mapByItem.containsKey(m_itemsIndex.get(((_Review)d).getItemID()))){//item coldstart
                    m_likelihood_array[2] += loglikelihood;
                    m_perplexity_array[2] += loglikelihood;
                    m_totalWords_array[2] += d.getTotalDocLength();
                    m_docSize_array[2] += 1;
                }else {
                    m_likelihood_array[3] += loglikelihood;
                    m_perplexity_array[3] += loglikelihood;
                    m_totalWords_array[3] += d.getTotalDocLength();
                    m_docSize_array[3] += 1;
                }

                m_likelihood_array[4] += loglikelihood;
                m_perplexity_array[4] += loglikelihood;
                m_totalWords_array[4] += d.getTotalDocLength();
                m_docSize_array[4] += 1;
            }

        }
        System.out.format("[Stat]Test evaluation finished: %d docs\n", m_testSet.size());

        //0,1: perplexity_coldstart_all, likelihood_coldstart_all
        //2,3: perplexity_coldstart_user, likelihood_coldstart_user
        //4,5: perplexity_coldstart_item, likelihood_coldstart_item
        //6,7: perplexity_normal, likelihood_normal
        //8,9: perplexity, likelihood
        double[] results = new double[10];
        for(int i = 0; i < 5; i++){
            if(m_totalWords_array[i] > 0){
                results[2*i] = Math.exp(- m_perplexity_array[i] / m_totalWords_array[i]);
                results[2*i+1] = m_likelihood_array[i] / m_docSize_array[i];
            }
            System.out.format("[Stat]%d part has %f docs: perplexity is %.3f and log-likelihood is %.3f\n",
                    i, m_docSize_array[i], results[2*i], results[2*i+1]);
        }
        return results;
    }

    //k-fold Cross Validation.
    @Override
    public void crossValidation(int k) {
        analyzeCorpus();
        m_trainSet = new ArrayList<_Doc>();
        m_testSet = new ArrayList<_Doc>();

        double[] perf = new double[k];
        double[] like = new double[k];
        System.out.println("[Info]Start RANDOM cross validation...");
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

                System.out.format("====================\n[Info]Fold No. %d: train size = %d, test size = %d....\n", i, m_trainSet.size(), m_testSet.size());

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
                double[] results = Evaluation2();
                perf[i] = results[0];
                like[i] = results[1];

                System.out.format("[Info]%s Train/Test finished in %.2f seconds...\n", this.toString(), (System.currentTimeMillis()-start)/1000.0);
                m_trainSet.clear();
                m_testSet.clear();
            }
        }

        //output the performance statistics
        double mean = Utils.sumOfArray(perf)/k, var = 0;
        for(int i=0; i<perf.length; i++)
            var += (perf[i]-mean) * (perf[i]-mean);
        var = Math.sqrt(var/k);
        System.out.format("[Stat]Perplexity %.3f+/-%.3f\n", mean, var);

        mean = Utils.sumOfArray(like)/k;
        var = 0;
        for(int i=0; i<like.length; i++)
            var += (like[i]-mean) * (like[i]-mean);
        var = Math.sqrt(var/k);
        System.out.format("[Stat]Loglikelihood %.3f+/-%.3f\n", mean, var);
    }

    @Override
    public void printParameterAggregation(int k, String folderName, String topicmodel) {
        super.printParameterAggregation(k, folderName, topicmodel);

        String alphaPath = String.format("%s%s_alphaBy%s_%d.txt", folderName, topicmodel, m_mode.equals("User")?"User":"Item", number_of_topics);


    }

    @Override
    public void printParam(String folderName, String topicmodel){
        //print out prior parameter of dirichlet: alpha
        File file = new File(folderName);
        try{
            file.getParentFile().mkdirs();
            file.createNewFile();
        } catch(IOException e){
            e.printStackTrace();
        }
        try{
            PrintWriter alphaWriter = new PrintWriter(file);
            alphaWriter.format("%d\t%d\n", docCluster.size(), number_of_topics);

            for (int i = 0; i < number_of_topics; i++)
                alphaWriter.format("%.5f\t", this.m_alpha[i]);
            alphaWriter.close();
        } catch(FileNotFoundException ex){
            System.err.format("[Error]Failed to open file %s\n", folderName);
        }
    }

}
