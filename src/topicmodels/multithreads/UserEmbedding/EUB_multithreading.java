package topicmodels.multithreads.UserEmbedding;

import structures.*;
import topicmodels.UserEmbedding.EUB;
import topicmodels.multithreads.EmbedModelWorker;
import topicmodels.multithreads.EmbedModel_worker;
import topicmodels.multithreads.TopicModelWorker;
import topicmodels.multithreads.TopicModel_worker;

import java.util.Collection;

/**
 * @author Lin Gong (lg5bt@virginia.edu)
 */
public class EUB_multithreading extends EUB {

    protected EmbedModelWorker[] m_userWorkers = null;
    protected EmbedModelWorker[] m_topicWorkers = null;

    public class Doc_worker extends TopicModel_worker {
        public Doc_worker(int number_of_topics, int vocabulary_size){
            super(number_of_topics, vocabulary_size);
        }

        @Override
        public void run(){
            m_likelihood = 0;
            double loglikelihood;
            for(_Doc d: m_corpus) {
                if (m_type == TopicModel_worker.RunType.RT_EM)
                    m_likelihood += calculate_E_step(d);
                else if (m_type == TopicModel_worker.RunType.RT_inference) {
                    loglikelihood = inference(d);
                }
            }
        }

        @Override
        public double calculate_E_step(_Doc d) {
            _Doc4EUB doc = (_Doc4EUB) d;
            double cur = varInference4Doc(doc);
            updateStats4Doc(doc);
            return cur;
        }

        @Override
        public double inference(_Doc d){
            initTestDoc(d);
            double likelihood = calculate_E_step(d);
            estThetaInDoc(d);
            return likelihood;
        }

        @Override
        public double accumluateStats(double[][] word_topic_sstat) {
            return super.accumluateStats(word_topic_sstat);
        }

    }

    public class User_worker extends EmbedModel_worker {
        protected double m_likelihood_perplexity;

        public User_worker(int dim){
            super(dim);
        }

        @Override
        public void run(){
            m_likelihood = 0;
            double loglikelihood = 0;
            for(Object o: m_objects){
                _User4EUB user = (_User4EUB) o;
                if (m_type == TopicModel_worker.RunType.RT_EM)
                    m_likelihood += calculate_E_step(user);
                else if (m_type == TopicModel_worker.RunType.RT_inference) {
                    loglikelihood = varInference4User(user);
                }
            }
        }

        @Override
        public double calculate_E_step(Object o) {
            _User4EUB user = (_User4EUB) o;
            return varInference4User(user);
        }

        @Override
        public double accumluateStats(){
            return m_likelihood;
        }

        @Override
        public void resetStats(){}
    }

    public class Topic_worker extends EmbedModel_worker{

        public Topic_worker(int dim){
            super(dim);
        }

        @Override
        public void run(){
            m_likelihood = 0;
            double loglikelihood = 0;

            for(Object o: m_objects){
                _Topic4EUB topic = (_Topic4EUB) o;
                if (m_type == TopicModel_worker.RunType.RT_EM)
                    m_likelihood += calculate_E_step(topic);
                else if (m_type == TopicModel_worker.RunType.RT_inference) {
                    loglikelihood = varInference4Topic(topic);
                }
            }
        }

        @Override
        public double calculate_E_step(Object o) {
            _Topic4EUB topic = (_Topic4EUB) o;
            return varInference4Topic(topic);
        }

        @Override
        public double accumluateStats(){
            return m_likelihood;
        }

        @Override
        public void resetStats(){}

    }

    public EUB_multithreading(int number_of_iteration, double converge, double beta, _Corpus c, double lambda,
                              int number_of_topics, double alpha, int varMaxIter, double varConverge, int m) {
        super(number_of_iteration, converge, beta, c, lambda, number_of_topics, alpha, varMaxIter, varConverge, m);
        m_multithread = true;
    }

    protected void initialize_probability(Collection<_Doc> collection) {
        int cores = Runtime.getRuntime().availableProcessors();
        m_threadpool = new Thread[cores];
        m_workers = new TopicModel_worker[cores];
        m_topicWorkers = new EUB_multithreading.Topic_worker[cores];
        m_userWorkers = new EUB_multithreading.User_worker[cores];

        for(int i=0; i<cores; i++) {
            m_workers[i] = new EUB_multithreading.Doc_worker(number_of_topics, vocabulary_size);
            m_topicWorkers[i] = new EUB_multithreading.Topic_worker(m_embeddingDim);
            m_userWorkers[i] = new EUB_multithreading.User_worker(m_embeddingDim);
        }

        int workerID = 0;
        for(_Doc d: collection) {
            m_workers[workerID%cores].addDoc(d);
            workerID++;
        }
        workerID = 0;
        for(_Topic4EUB t: m_topics){
            m_topicWorkers[workerID%cores].addObject(t);
            workerID++;
        }
        workerID = 0;
        for(_User4EUB u: m_users) {
            m_userWorkers[workerID%cores].addObject(u);
            workerID++;
        }

        super.initialize_probability(collection);
    }


    @Override
    public double multithread_E_step() {
        int iter = 0;
        double likelihood , last = -1.0, converge;

        do {
            init();

            //doc
            likelihood = super.multithread_E_step();

            if(Double.isNaN(likelihood) || Double.isInfinite(likelihood)){
                System.err.println("[Error]E_step for document produces NaN likelihood...");
                break;
            }

            //users
            likelihood += multithread_general(m_topicWorkers);

            if(Double.isNaN(likelihood) || Double.isInfinite(likelihood)){
                System.err.println("[Error]E_step for user produces NaN likelihood...");
                break;
            }

            //items
            likelihood += multithread_general(m_userWorkers);

            if(Double.isNaN(likelihood) || Double.isInfinite(likelihood)){
                System.err.println("[Error]E_step for item produces NaN likelihood...");
                break;
            }

            if(iter > 0)
                converge = Math.abs((likelihood - last) / last);
            else
                converge = 1.0;

            last = likelihood;

            if(converge < m_varConverge)
                break;
            if(iter % 10 == 0)
                System.out.format("[Info]Multi-thread E-Step: %d iteration, likelihood=%.2f, converge to %.8f\n",
                        iter, last, converge);
        }while(iter++ < m_varMaxIter);
        System.out.print(String.format("[Info]Finish E-Step: %d iteration, likelihood: %.4f\n", iter, likelihood));

        return likelihood;
    }


    protected double multithread_general(EmbedModelWorker[] workers){
        double likelihood = 0.0;
        for (int i = 0; i < workers.length; i++) {
            workers[i].setType(TopicModel_worker.RunType.RT_EM);
            m_threadpool[i] = new Thread(workers[i]);
            m_threadpool[i].start();
        }
        for (Thread thread : m_threadpool) {
            try {
                thread.join();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
        for (EmbedModelWorker worker : workers)
            likelihood += worker.accumluateStats();

        return likelihood;
    }


    @Override
    protected double multithread_inference() {
        int iter = 0;
        double likelihood, last = -1.0, converge;

        //clear up for adding new testing documents
        for (int i = 0; i < m_workers.length; i++) {
            m_workers[i].setType(TopicModel_worker.RunType.RT_inference);
            m_workers[i].clearCorpus();
        }

        //evenly allocate the testing work load
        int workerID = 0;
        for (_Doc d : m_testSet) {
            m_workers[workerID % m_workers.length].addDoc(d);
            workerID++;
        }

        do {
            init();
            likelihood = 0.0;

            //run
            for (int i = 0; i < m_workers.length; i++) {
                m_threadpool[i] = new Thread(m_workers[i]);
                m_threadpool[i].start();
            }

            //wait till all finished
            for (Thread thread : m_threadpool) {
                try {
                    thread.join();
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            }

            double loglikelihood = 0, totalWords = 0, docSize = 0;
            for (TopicModelWorker worker : m_workers) {
                loglikelihood += ((Doc_worker) worker).getLogLikelihood();
                totalWords += ((Doc_worker) worker).getTotalWords();

            }

            if(Double.isNaN(likelihood) || Double.isInfinite(likelihood)){
                System.err.format("[Error]Inference generate NaN\n");
                break;
            }

            if(iter > 0)
                converge = Math.abs((likelihood - last) / last);
            else
                converge = 1.0;

            last = likelihood;
            if(converge < m_varConverge)
                break;
            System.out.print("---likelihood: " + last + "\n");
        }while(iter++ < m_varMaxIter);

        System.out.print(String.format("[Info]Inference finished: likelihood: %.4f\n", likelihood));

        return likelihood;
    }

}
