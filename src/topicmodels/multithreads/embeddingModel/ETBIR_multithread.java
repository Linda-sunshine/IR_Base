package topicmodels.multithreads.embeddingModel;

import structures.*;
import topicmodels.TopicModel;
import topicmodels.embeddingModel.ETBIR;
import topicmodels.multithreads.EmbedModelWorker;
import topicmodels.multithreads.EmbedModel_worker;
import topicmodels.multithreads.LDA.LDA_Variational_multithread;
import topicmodels.multithreads.TopicModelWorker;
import topicmodels.multithreads.TopicModel_worker;
import utils.Utils;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;

public class ETBIR_multithread extends ETBIR {
    protected Thread[] m_userThreadpool = null;
    protected EmbedModelWorker[] m_userWorkers = null;
    protected Thread[] m_itemThreadpool = null;
    protected EmbedModelWorker[] m_itemWorkers = null;

    public class Doc_worker extends TopicModel_worker {
        protected double thetaStats;
        protected double eta_mean_Stats;
        protected double eta_p_Stats;

        public Doc_worker(int number_of_topics, int vocabulary_size) {
            super(number_of_topics, vocabulary_size);
        }

        @Override
        public void run() {
            m_likelihood = 0;
            m_perplexity = 0;

            double loglikelihood = 0, log2 = Math.log(2.0);
            long eStartTime = System.currentTimeMillis();

            for(_Doc d:m_corpus) {
                if (m_type == RunType.RT_EM)
                    m_likelihood += calculate_E_step(d);
                else if (m_type == RunType.RT_inference) {
                    loglikelihood = inference(d);
                    m_perplexity += Math.pow(2.0, -loglikelihood/d.getTotalDocLength() / log2);
                    m_likelihood += loglikelihood;
                }
            }
            long eEndTime = System.currentTimeMillis();

            // System.out.println("per thread per iteration e step time\t"
            // + (eEndTime - eStartTime));

        }

        @Override
        public double calculate_E_step(_Doc d) {
            _Doc4ETBIR doc = (_Doc4ETBIR)d;

            String userID = doc.getTitle();
            String itemID = doc.getItemID();
            _User4ETBIR currentU = m_users.get(m_usersIndex.get(userID));
            _Product4ETBIR currentI = m_items.get(m_itemsIndex.get(itemID));

            double cur = varInference4Doc(doc, currentU, currentI);
            updateStats4Doc(doc);
            return cur;
        }

        protected void updateStats4Doc(_Doc d){
            _Doc4ETBIR doc = (_Doc4ETBIR) d;
            // update m_word_topic_stats for updating beta
            _SparseFeature[] fv = doc.getSparse();
            int wid;
            double v;
            for(int n=0; n<fv.length; n++) {
                wid = fv[n].getIndex();
                v = fv[n].getValue();
                for(int i=0; i<number_of_topics; i++)
                    sstat[i][wid] += v*doc.m_phi[n][i];
            }

            // update m_thetaStats for updating rho
            for(int k = 0; k < number_of_topics; k++)
                thetaStats += doc.m_Sigma[k] + doc.m_mu[k] * doc.m_mu[k];

            // update m_eta_p_stats for updating rho
            // update m_eta_mean_stats for updating rho
            double eta_mean_temp = 0.0;
            double eta_p_temp = 0.0;
            _Product4ETBIR item = m_items.get(m_itemsIndex.get(doc.getItemID()));
            _User4ETBIR user = m_users.get(m_usersIndex.get(doc.getTitle()));
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
            eta_mean_Stats += eta_mean_temp / eta0;
            eta_p_Stats += eta_p_temp / (eta0 * (eta0 + 1.0));
        }

        @Override
        public double accumluateStats(double[][] word_topic_sstat) {
            m_thetaStats += thetaStats;
            m_eta_mean_Stats += eta_mean_Stats;
            m_eta_p_Stats += eta_p_Stats;

            return super.accumluateStats(word_topic_sstat);
        }

        @Override
        public void resetStats() {
            m_thetaStats = 0.0;
            m_eta_p_Stats = 0.0;
            m_eta_mean_Stats = 0.0;
            super.resetStats();
        }

        @Override
        public double inference(_Doc d) {
            initTestDoc(d);
            double likelihood = calculate_E_step(d);
            estThetaInDoc(d);
            return likelihood;
        }
    }

    public class Item_worker extends EmbedModel_worker {
        protected double[] alphaStat;

        public Item_worker(int number_of_topics, int vocabulary_size) {
            super(number_of_topics, vocabulary_size);
        }

        @Override
        public void run() {
            m_likelihood = 0;
            m_perplexity = 0;

            double loglikelihood = 0, log2 = Math.log(2.0);
            long eStartTime = System.currentTimeMillis();

            for (Object o : m_objects) {
                _Product4ETBIR i = (_Product4ETBIR) o;
                if (m_type == TopicModel_worker.RunType.RT_EM)
                    m_likelihood += calculate_E_step(i);
                else if (m_type == TopicModel_worker.RunType.RT_inference) {

                }
            }
            long eEndTime = System.currentTimeMillis();
        }

        @Override
        public double calculate_E_step(Object o) {
            _Product4ETBIR i = (_Product4ETBIR) o;
            double cur = varInference4Item(i);
            updateStats4Item(i);
            return cur;
        }

        protected void updateStats4Item(_Product4ETBIR item){
            double digammaSum = Utils.digamma(Utils.sumOfArray(item.m_eta));
            for(int k = 0; k < number_of_topics; k++)
                alphaStat[k] += Utils.digamma(item.m_eta[k]) - digammaSum;
        }

        @Override
        public double accumluateStats() {
            for(int k=0; k<number_of_topics; k++)
                m_alphaStat[k] += alphaStat[k];

            return m_likelihood;
        }

        @Override
        public void resetStats() {
            Arrays.fill(alphaStat, 0);
        }
    }

    public class User_worker extends EmbedModel_worker {
        double pStats;

        public User_worker(int number_of_topics, int vocabulary_size) {
            super(number_of_topics, vocabulary_size);
        }

        @Override
        public void run() {
            m_likelihood = 0;
            m_perplexity = 0;

            double loglikelihood = 0, log2 = Math.log(2.0);
            long eStartTime = System.currentTimeMillis();

            for (Object o : m_objects) {
                _User4ETBIR u = (_User4ETBIR) o;
                if (m_type == TopicModel_worker.RunType.RT_EM)
                    m_likelihood += calculate_E_step(u);
                else if (m_type == TopicModel_worker.RunType.RT_inference) {

                }
            }
            long eEndTime = System.currentTimeMillis();
        }


        @Override
        public double calculate_E_step(Object o) {
            _User4ETBIR u = (_User4ETBIR) o;
            double cur = varInference4User(u);
            updateStats4User(u);
            return cur;
        }

        protected void updateStats4User(_User4ETBIR user){
            for(int k = 0; k < number_of_topics; k++){
                for(int l = 0; l < number_of_topics; l++){
                    pStats += user.m_SigmaP[k][l][l] + user.m_nuP[k][l] * user.m_nuP[k][l];
                }
            }
        }

        @Override
        public double accumluateStats() {
            m_pStats += pStats;
            return m_likelihood;
        }

        @Override
        public void resetStats() {
            pStats = 0.0;
        }

    }

    public ETBIR_multithread(int emMaxIter, double emConverge,
                             double beta, _Corpus corpus, double lambda,
                             int number_of_topics, double alpha, int varMaxIter, double varConverge, //LDA_variational
                             double sigma, double rho){
        super(emMaxIter, emConverge, beta, corpus, lambda,
                number_of_topics, alpha, varMaxIter, varConverge, sigma, rho);
        m_multithread = true;
    }

    protected void initialize_probability(Collection<_Doc> collection, Collection<_User4ETBIR> users,
                                          Collection<_Product4ETBIR> items) {
        int cores = Runtime.getRuntime().availableProcessors();
        m_threadpool = new Thread[cores];
        m_workers = new ETBIR_multithread.Doc_worker[cores];

        m_itemThreadpool = new Thread[cores];
        m_itemWorkers = new ETBIR_multithread.Item_worker[cores];

        m_userThreadpool = new Thread[cores];
        m_userWorkers = new ETBIR_multithread.User_worker[cores];

        for(int i=0; i<cores; i++) {
            m_workers[i] = new ETBIR_multithread.Doc_worker(number_of_topics, vocabulary_size);
            m_itemWorkers[i] = new ETBIR_multithread.Item_worker(number_of_topics, vocabulary_size);
            m_userWorkers[i] = new ETBIR_multithread.User_worker(number_of_topics, vocabulary_size);
        }

        int workerID = 0;
        for(_Doc d:collection) {
            m_workers[workerID%cores].addDoc(d);
            workerID++;
        }
        workerID = 0;
        for(_Product4ETBIR i:items){
            m_itemWorkers[workerID%cores].addObject(i);
            workerID++;
        }
        workerID = 0;
        for(_User4ETBIR u:users){
            m_userWorkers[workerID%cores].addObject(u);
            workerID++;
        }

        super.initialize_probability(collection, users, items);
    }


    @Override
    protected void init() { // clear up for next iteration
        super.init();
        for(TopicModelWorker worker:m_workers)
            worker.resetStats();
        for(EmbedModelWorker worker:m_itemWorkers)
            worker.resetStats();
        for(EmbedModelWorker worker:m_userWorkers)
            worker.resetStats();
    }

    @Override
    public double multithread_E_step() {
        //doc
        double likelihood = super.multithread_E_step();

        //users
        for(int i = 0; i < m_userWorkers.length; i++){
            m_userWorkers[i].setType(TopicModel_worker.RunType.RT_EM);
            m_userThreadpool[i] = new Thread(m_userWorkers[i]);
            m_userThreadpool[i].start();
        }
        for(Thread thread:m_userThreadpool){
            try {
                thread.join();
            }catch (InterruptedException e){
                e.printStackTrace();
            }
        }
        for(EmbedModelWorker worker:m_userWorkers){
            likelihood += worker.accumluateStats();
        }

        //items
        for(int i = 0; i < m_itemWorkers.length; i++){
            m_itemWorkers[i].setType(TopicModel_worker.RunType.RT_EM);
            m_itemThreadpool[i] = new Thread(m_itemThreadpool[i]);
            m_itemThreadpool[i].start();
        }
        for(Thread thread:m_itemThreadpool){
            try {
                thread.join();
            }catch (InterruptedException e){
                e.printStackTrace();
            }
        }
        for(EmbedModelWorker worker:m_itemWorkers){
            likelihood += worker.accumluateStats();
        }

        return likelihood;
    }
}
