package topicmodels.multithreads.embeddingModel;

import structures.*;
import topicmodels.embeddingModel.ETBIR;
import topicmodels.multithreads.LDA.LDA_Variational_multithread;
import topicmodels.multithreads.TopicModelWorker;
import topicmodels.multithreads.TopicModel_worker;

import java.util.Collection;

public class ETBIR_multithread extends ETBIR {
    protected Thread[] m_userThreadpool = null;
    protected TopicModelWorker[] m_userWorkers = null;
    protected Thread[] m_itemThreadpool = null;
    protected TopicModelWorker[] m_itemWorkers = null;


    public class ETBIR_worker extends TopicModel_worker {
        protected double[] alphaStat;

        public ETBIR_worker(int number_of_topics, int vocabulary_size) {
            super(number_of_topics, vocabulary_size);
            alphaStat = new double[number_of_topics];
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


    }

    public ETBIR_multithread(int emMaxIter, double emConverge,
                             double beta, _Corpus corpus, double lambda,
                             int number_of_topics, double alpha, int varMaxIter, double varConverge, //LDA_variational
                             double sigma, double rho){
        super(emMaxIter, emConverge, beta, corpus, lambda,
                number_of_topics, alpha, varMaxIter, varConverge, sigma, rho);
        m_multithread = true;
    }

    @Override
    protected void initialize_probability(Collection<_Doc> collection, Collection<_User> users, Collection<_Product> items) {
        int cores = Runtime.getRuntime().availableProcessors();
        m_threadpool = new Thread[cores];
        m_workers = new LDA_Variational_multithread.LDA_worker[cores];

        for(int i=0; i<cores; i++)
            m_workers[i] = new LDA_Variational_multithread.LDA_worker(number_of_topics, vocabulary_size);

        int workerID = 0;
        for(_Doc d:collection) {
            m_workers[workerID%cores].addDoc(d);
            workerID++;
        }

        super.initialize_probability(collection);
    }

    @Override
    public double multithread_E_step() {
        for(int i=0; i<m_workers.length; i++) {
            m_workers[i].setType(TopicModel_worker.RunType.RT_EM);
            m_threadpool[i] = new Thread(m_workers[i]);
            m_threadpool[i].start();
        }

        //wait till all finished
        for(Thread thread:m_threadpool){
            try {
                thread.join();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }

        double likelihood = 0;
        for(TopicModelWorker worker:m_workers)
            likelihood += worker.accumluateStats(word_topic_sstat);
        return likelihood;
    }
}
