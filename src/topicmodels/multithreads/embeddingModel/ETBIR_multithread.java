package topicmodels.multithreads.embeddingModel;

import structures.*;
import topicmodels.TopicModel;
import topicmodels.embeddingModel.ETBIR;
import topicmodels.multithreads.EmbedModelWorker;
import topicmodels.multithreads.EmbedModel_worker;
import topicmodels.multithreads.LDA.LDA_Focus_multithread;
import topicmodels.multithreads.LDA.LDA_Variational_multithread;
import topicmodels.multithreads.TopicModelWorker;
import topicmodels.multithreads.TopicModel_worker;
import utils.Utils;

import javax.jws.Oneway;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;

public class ETBIR_multithread extends ETBIR {
    protected EmbedModelWorker[] m_userWorkers = null;
    protected EmbedModelWorker[] m_itemWorkers = null;

    public class Doc_worker extends TopicModel_worker {
        //coldstart_all, coldstart_user, coldstart_item, normal;
        protected double[] likelihood_array;
        protected double[] perplexity_array;
        protected double[] totalWords_array;
        protected double[] docSize_array;

        protected double thetaStats;
        protected double eta_mean_Stats;
        protected double eta_p_Stats;

        public Doc_worker(int number_of_topics, int vocabulary_size) {
            super(number_of_topics, vocabulary_size);
            likelihood_array = new double[5];
            perplexity_array = new double[5];
            totalWords_array = new double[5];
            docSize_array = new double[5];
        }

        public double[] getLogLikelihoodArray() {
            return likelihood_array;
        }

        public double[] getPerplexityArray() {
            return perplexity_array;
        }

        public double[] getTotalWordsArray(){
            return totalWords_array;
        }

        public double[] getDocSizeArray(){
            return docSize_array;
        }

        @Override
        public void run() {
            m_likelihood = 0;
            m_perplexity = 0;
            m_totalWords = 0;
            Arrays.fill(likelihood_array, 0);
            Arrays.fill(perplexity_array, 0);
            Arrays.fill(totalWords_array, 0);
            Arrays.fill(docSize_array, 0);

            double loglikelihood = 0, log2 = Math.log(2.0);
            // System.out.println("thread corpus size\t" + m_corpus.size());
            long eStartTime = System.currentTimeMillis();

            for(_Doc d:m_corpus) {
                if (m_type == TopicModel_worker.RunType.RT_EM)
                    m_likelihood += calculate_E_step(d);
                else if (m_type == TopicModel_worker.RunType.RT_inference) {
                    loglikelihood = inference(d);

                    if(!m_mapByUser.containsKey(m_usersIndex.get(((_Review)d).getUserID()))
                            && !m_mapByItem.containsKey(m_itemsIndex.get(((_Review)d).getItemID()))){//all coldstart
                        likelihood_array[0] += loglikelihood;
                        perplexity_array[0] += loglikelihood;
                        totalWords_array[0] += d.getTotalDocLength();
                        docSize_array[0] += 1;
                    }else if(!m_mapByUser.containsKey(m_usersIndex.get(((_Review)d).getUserID()))
                            && m_mapByItem.containsKey(m_itemsIndex.get(((_Review)d).getItemID()))){//user coldstart
                        likelihood_array[1] += loglikelihood;
                        perplexity_array[1] += loglikelihood;
                        totalWords_array[1] += d.getTotalDocLength();
                        docSize_array[1] += 1;
                    }else if(m_mapByUser.containsKey(m_usersIndex.get(((_Review)d).getUserID()))
                            && !m_mapByItem.containsKey(m_itemsIndex.get(((_Review)d).getItemID()))){//item coldstart
                        likelihood_array[2] += loglikelihood;
                        perplexity_array[2] += loglikelihood;
                        totalWords_array[2] += d.getTotalDocLength();
                        docSize_array[2] += 1;
                    }else {
                        likelihood_array[3] += loglikelihood;
                        perplexity_array[3] += loglikelihood;
                        totalWords_array[3] += d.getTotalDocLength();
                        docSize_array[3] += 1;
                    }
//				m_perplexity += Math.pow(2.0, -loglikelihood/d.getTotalDocLength() / log2);
                    m_likelihood += loglikelihood;
                    m_perplexity += loglikelihood;
                    m_totalWords += d.getTotalDocLength();

                    likelihood_array[4] += loglikelihood;
                    perplexity_array[4] += loglikelihood;
                    totalWords_array[4] += d.getTotalDocLength();
                    docSize_array[4] += 1;

                }
            }
            long eEndTime = System.currentTimeMillis();

            // System.out.println("per thread per iteration e step time\t"
            // + (eEndTime - eStartTime));

        }

        @Override
        public double calculate_E_step(_Doc d) {
            _Doc4ETBIR doc = (_Doc4ETBIR)d;

            String userID = doc.getUserID();
            String itemID = doc.getItemID();
            _User4ETBIR currentU = (_User4ETBIR) m_users.get(m_usersIndex.get(userID));
            _Product4ETBIR currentI = (_Product4ETBIR) m_items.get(m_itemsIndex.get(itemID));

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
            thetaStats = 0.0;
            eta_p_Stats = 0.0;
            eta_mean_Stats = 0.0;
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
        protected double[] likelihood_array;
        protected double[] perplexity_array;

        protected double[] alphaStat;

        public Item_worker(int number_of_topics, int vocabulary_size) {
            super(number_of_topics, vocabulary_size);
            alphaStat = new double[number_of_topics];
            likelihood_array = new double[3];
            perplexity_array = new double[3];
        }

        public double[] getLogLikelihoodArray() { return likelihood_array; }

        public double[] getPerplexityArray() {
            return perplexity_array;
        }

        @Override
        public void run() {
            m_likelihood = 0;
            m_perplexity = 0;
            Arrays.fill(likelihood_array, 0);
            Arrays.fill(perplexity_array, 0);

            double loglikelihood = 0;
            for (Object o : m_objects) {
                _Product4ETBIR i = (_Product4ETBIR) o;
                if (m_type == TopicModel_worker.RunType.RT_EM)
                    m_likelihood += calculate_E_step(i);
                else if (m_type == TopicModel_worker.RunType.RT_inference) {
                    loglikelihood = inference(i);

                    if(!m_mapByItem.containsKey(m_itemsIndex.get(i.getID()))){//item coldstart
                        likelihood_array[0] += loglikelihood;
                        perplexity_array[0] += loglikelihood;
                    }else {
                        likelihood_array[1] += loglikelihood;
                        perplexity_array[1] += loglikelihood;

                    }

                    likelihood_array[2] += loglikelihood;
                    perplexity_array[2] += loglikelihood;
                    m_likelihood += loglikelihood;
                    m_perplexity += loglikelihood;
                }
            }
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
        protected double[] likelihood_array;
        protected double[] perplexity_array;

        double pStats;
        double lambda_Stats;

        public User_worker(int number_of_topics, int vocabulary_size) {
            super(number_of_topics, vocabulary_size);
            likelihood_array = new double[3];
            perplexity_array = new double[3];
        }

        public double[] getLogLikelihoodArray() {
            return likelihood_array;
        }

        public double[] getPerplexityArray() {
            return perplexity_array;
        }

        @Override
        public void run() {
            m_likelihood = 0;
            m_perplexity = 0;
            Arrays.fill(likelihood_array, 0);
            Arrays.fill(perplexity_array, 0);

            double loglikelihood = 0.0;
            for (Object o : m_objects) {
                _User4ETBIR u = (_User4ETBIR) o;
                if (m_type == TopicModel_worker.RunType.RT_EM)
                    m_likelihood += calculate_E_step(u);
                else if (m_type == TopicModel_worker.RunType.RT_inference) {
                    loglikelihood = inference(u);

                    if(!m_mapByUser.containsKey(m_usersIndex.get(u.getUserID()))){//user coldstart
                        likelihood_array[0] += loglikelihood;
                        perplexity_array[0] += loglikelihood;
                    }else {
                        likelihood_array[1] += loglikelihood;
                        perplexity_array[1] += loglikelihood;

                    }

                    likelihood_array[2] += loglikelihood;
                    perplexity_array[2] += loglikelihood;
                    m_likelihood += loglikelihood;
                    m_perplexity += loglikelihood;
                }
            }
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
                    if(!m_flag_diagonal_lambda)
                        pStats += user.m_SigmaP[k][l][l] + user.m_nuP[k][l] * user.m_nuP[k][l]
                                - 2 * m_lambda * Utils.sumOfArray(user.m_nuP[k]) + m_lambda * m_lambda * number_of_topics;
                    else
                        pStats += user.m_SigmaP[k][l][l] + user.m_nuP[k][l] * user.m_nuP[k][l] - 2 * m_lambda * user.m_nuP[k][k] + m_lambda * m_lambda;
                }
                if(!m_flag_diagonal_lambda)
                    lambda_Stats += Utils.sumOfArray(user.m_nuP[k]);
                else
                    lambda_Stats += user.m_nuP[k][k];
            }
        }

        @Override
        public double accumluateStats() {
            m_pStats += pStats;
            m_lambda_Stats += lambda_Stats;
            return m_likelihood;
        }

        @Override
        public void resetStats() {
            pStats = 0.0;
            lambda_Stats = 0.0;
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

    protected void initialize_probability(Collection<_Doc> collection) {
        int cores = Runtime.getRuntime().availableProcessors();
        m_threadpool = new Thread[cores];
        m_workers = new ETBIR_multithread.Doc_worker[cores];
        m_itemWorkers = new ETBIR_multithread.Item_worker[cores];
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
        for(int i_idx : m_mapByItem.keySet()){
            Object item = m_items.get(i_idx);
            m_itemWorkers[workerID%cores].addObject(item);
            workerID++;
        }
        workerID = 0;
        for(int u_idx:m_mapByUser.keySet()) {
            Object user = m_users.get(u_idx);
            m_userWorkers[workerID%cores].addObject(user);
            workerID++;
        }

        super.initialize_probability(collection);
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
    public double multithread_E_step() {
        int iter = 0;
        double likelihood = 0.0, last = -1.0, converge = 0.0;

        boolean warning;
        do {
            init();
            warning = false;

            //doc
            likelihood = super.multithread_E_step();

            if(Double.isNaN(likelihood) || Double.isInfinite(likelihood)){
                System.err.println("[Error]E_step for document produces NaN likelihood...");
                warning = true;
                break;
            }

            //users
            likelihood += multithread_general(m_userWorkers);

            if(Double.isNaN(likelihood) || Double.isInfinite(likelihood)){
                System.err.println("[Error]E_step for user produces NaN likelihood...");
                warning = true;
                break;
            }

            //items
            likelihood += multithread_general(m_itemWorkers);

            if(Double.isNaN(likelihood) || Double.isInfinite(likelihood)){
                System.err.println("[Error]E_step for item produces NaN likelihood...");
                warning = true;
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
        }while(iter++ < m_varMaxIter && !warning);
        System.out.print(String.format("[Info]Finish E-Step: %d iteration, likelihood: %.4f\n", iter, likelihood));

        return likelihood;
    }

    @Override
    protected double multithread_inference() {
        int iter = 0;
        double likelihood = 0.0, likelihood_doc = 0.0, last = -1.0, converge = 0.0;
        boolean warning;

        //clear up for adding new testing documents
        for(int i=0; i<m_workers.length; i++) {
            m_workers[i].setType(TopicModel_worker.RunType.RT_inference);
            m_workers[i].clearCorpus();
        }
        for(int i = 0; i < m_itemWorkers.length; i++){
            m_itemWorkers[i].setType(TopicModel_worker.RunType.RT_inference);
            m_itemWorkers[i].clearObjects();
        }
        for(int i = 0;i < m_userWorkers.length; i++){
            m_userWorkers[i].setType(TopicModel_worker.RunType.RT_inference);
            m_userWorkers[i].clearObjects();
        }

        //evenly allocate the testing work load
        int workerID = 0;
        for(_Doc d:m_testSet) {
            m_workers[workerID % m_workers.length].addDoc(d);
            workerID++;
        }
        workerID = 0;
        for(int i_idx:m_mapByItem_test.keySet()){
            _Product4ETBIR i = (_Product4ETBIR) m_items.get(i_idx);
            m_itemWorkers[workerID%m_itemWorkers.length].addObject(i);
            workerID++;
        }
        workerID = 0;
        for(int u_idx:m_mapByUser_test.keySet()){
            _User4ETBIR u = (_User4ETBIR) m_users.get(u_idx);
            m_userWorkers[workerID%m_userWorkers.length].addObject(u);
            workerID++;
        }

        do {
            init();
            likelihood = 0.0;
            Arrays.fill(m_likelihood_array, 0);
            Arrays.fill(m_perplexity_array, 0);
            Arrays.fill(m_totalWords_array,0);
            Arrays.fill(m_docSize_array, 0);
            
            warning = false;
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

            for (int i = 0; i < m_userWorkers.length; i++) {
                m_threadpool[i] = new Thread(m_userWorkers[i]);
                m_threadpool[i].start();
            }
            for (Thread thread : m_threadpool) {
                try {
                    thread.join();
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            }

            for (int i = 0; i < m_itemWorkers.length; i++) {
                m_threadpool[i] = new Thread(m_itemWorkers[i]);
                m_threadpool[i].start();
            }
            for (Thread thread : m_threadpool) {
                try {
                    thread.join();
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            }

            for (TopicModelWorker worker : m_workers) {
                Utils.add2Array(m_likelihood_array, ((Doc_worker) worker).getLogLikelihoodArray(), 1);
                Utils.add2Array(m_perplexity_array, ((Doc_worker) worker).getPerplexityArray(), 1);
                Utils.add2Array(m_totalWords_array, ((Doc_worker) worker).getTotalWordsArray(), 1);
                Utils.add2Array(m_docSize_array, ((Doc_worker) worker).getDocSizeArray(), 1);
                likelihood += worker.getLogLikelihood();
            }
            for (EmbedModelWorker worker : m_userWorkers) {
                m_likelihood_array[0] += (((User_worker)worker).getLogLikelihoodArray())[0];
                m_likelihood_array[1] += (((User_worker)worker).getLogLikelihoodArray())[0];
                m_likelihood_array[2] += (((User_worker)worker).getLogLikelihoodArray())[1];
                m_likelihood_array[3] += (((User_worker)worker).getLogLikelihoodArray())[1];
                m_likelihood_array[4] += (((User_worker)worker).getLogLikelihoodArray())[2];

                m_perplexity_array[0] += (((User_worker)worker).getPerplexityArray())[0];
                m_perplexity_array[1] += (((User_worker)worker).getPerplexityArray())[0];
                m_perplexity_array[2] += (((User_worker)worker).getPerplexityArray())[1];
                m_perplexity_array[3] += (((User_worker)worker).getPerplexityArray())[1];
                m_perplexity_array[4] += (((User_worker)worker).getPerplexityArray())[2];
                likelihood += worker.getLogLikelihood();
            }
            for (EmbedModelWorker worker : m_itemWorkers) {
                m_likelihood_array[0] += (((Item_worker)worker).getLogLikelihoodArray())[0];
                m_likelihood_array[1] += (((Item_worker)worker).getLogLikelihoodArray())[1];
                m_likelihood_array[2] += (((Item_worker)worker).getLogLikelihoodArray())[0];
                m_likelihood_array[3] += (((Item_worker)worker).getLogLikelihoodArray())[1];
                m_likelihood_array[4] += (((Item_worker)worker).getLogLikelihoodArray())[2];

                m_perplexity_array[0] += (((Item_worker)worker).getPerplexityArray())[0];
                m_perplexity_array[1] += (((Item_worker)worker).getPerplexityArray())[1];
                m_perplexity_array[2] += (((Item_worker)worker).getPerplexityArray())[0];
                m_perplexity_array[3] += (((Item_worker)worker).getPerplexityArray())[1];
                m_perplexity_array[4] += (((Item_worker)worker).getPerplexityArray())[2];
                likelihood += worker.getLogLikelihood();
            }

            if(Double.isNaN(likelihood) || Double.isInfinite(likelihood)){
                System.err.format("[Error]Inference generate NaN\n");
                warning = true;
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

        return likelihood; //only calculate document related likelihood
    }

}
