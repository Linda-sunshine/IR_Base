package topicmodels.multithreads.UserEmbedding;

import structures.*;
import topicmodels.multithreads.TopicModelWorker;
import topicmodels.multithreads.TopicModel_worker;

import java.lang.reflect.Array;
import java.util.ArrayList;

/**
 * @author Lin Gong (lg5bt@virginia.edu)
 */
public class EUB4ColdStart_multithreading extends EUB_multithreading {

    ArrayList<_Doc> m_testDocLight;
    ArrayList<_Doc> m_testDocMedium;
    ArrayList<_Doc> m_testDocHeavy;

    ArrayList<_User4EUB> m_testUserLight;
    ArrayList<_User4EUB> m_testUserMedium;
    ArrayList<_User4EUB> m_testUserHeavy;

    ArrayList<ArrayList<_Doc>> m_allTestDocs;
    ArrayList<ArrayList<_User4EUB>> m_allTestUsers;

    // the testSet in the TopicModel.java is for docs
    // the testUserSet is for inferring testing users
    ArrayList<_User4EUB> m_testUserSet;

    public EUB4ColdStart_multithreading(int number_of_iteration, double converge, double beta, _Corpus c, double lambda,
                                        int number_of_topics, double alpha, int varMaxIter, double varConverge, int m) {
        super(number_of_iteration, converge, beta, c, lambda, number_of_topics, alpha, varMaxIter, varConverge, m);
        m_testDocLight = new ArrayList<>();
        m_testDocMedium = new ArrayList<>();
        m_testDocHeavy = new ArrayList<>();
        m_allTestDocs = new ArrayList<>();

        m_testUserLight = new ArrayList<>();
        m_testUserMedium = new ArrayList<>();
        m_testUserHeavy = new ArrayList<>();
        m_allTestUsers = new ArrayList<>();
    }


    // fixed cross validation with specified fold number
    @Override
    public void fixedCrossValidation(int k, String saveDir){

        System.out.println(toString());

        double perplexity = 0;
        constructNetwork();

        System.out.format("\n==========Start %d-fold cross validation for cold start=========\n", k);
        m_trainSet.clear();
        m_testSet.clear();

        m_allTestDocs.clear();
        m_testDocLight.clear();
        m_testDocMedium.clear();
        m_testDocHeavy.clear();

        m_testUserLight.clear();
        m_testUserMedium.clear();
        m_testUserHeavy.clear();
        m_allTestUsers.clear();

        for(int i=0; i<m_users.size(); i++){
            _User4EUB user = m_users.get(i);
            for(_Review r: m_users.get(i).getReviews()){
                if(r.getMask4CV() == 0){
                    r.setType(_Doc.rType.TEST);
                    m_testDocLight.add(r);
                    if(!m_testUserLight.contains(user))
                        m_testUserLight.add(user);
                } else if(r.getMask4CV() == 1) {
                    r.setType(_Doc.rType.TEST);
                    m_testDocMedium.add(r);
                    if(!m_testUserMedium.contains(user))
                        m_testUserMedium.add(user);
                } else if(r.getMask4CV() == 2){
                    r.setType(_Doc.rType.TEST);
                    m_testDocHeavy.add(r);
                    if(!m_testUserHeavy.contains(user))
                        m_testUserHeavy.add(user);
                } else{
                    r.setType(_Doc.rType.TRAIN);
                    m_trainSet.add(r);
                }
            }
        }
        buildUserDocMap();
        EM();
        System.out.format("In one fold, (train: test_light: test_medium : test_heavy)=(%d : %d : %d : %d)\n",
                m_trainSet.size(), m_testDocLight.size(), m_testDocMedium.size(), m_testDocHeavy.size());
        if(m_mType == modelType.CV4DOC){
            System.out.println("[Info]Current mode is cv for docs, start evaluation....");
            m_allTestDocs.add(m_testDocLight);
            m_allTestDocs.add(m_testDocMedium);
            m_allTestDocs.add(m_testDocHeavy);
            double[] allPerplexity = new double[m_allTestDocs.size()];
            for(int i=0; i<m_allTestDocs.size(); i++) {
                m_testSet = m_allTestDocs.get(i);
                m_testUserSet = m_allTestUsers.get(i);

                perplexity = evaluation();
                allPerplexity[i] = perplexity;
            }
            System.out.println("================Perplexity===============");
            printPerplexity(allPerplexity);
        } else if(m_mType == modelType.CV4EDGE){
            System.out.println("[Info]Current mode is cv for edges, link predication is performed later.");
        } else{
            System.out.println("[error]Please specify the correct mode for evaluation!");
        }
        printStat4OneFold(k, saveDir, perplexity);
    }


    @Override
    protected double multithread_inference() {
        int iter = 0;
        double perplexity = 0, totalWords = 0, last = -1.0, converge;

        //clear up for adding new testing documents
        for (int i = 0; i < m_workers.length; i++) {
            m_workers[i].setType(TopicModel_worker.RunType.RT_inference);
            m_workers[i].clearCorpus();
        }

        for(int i = 0; i < m_topicWorkers.length; i++){
            m_topicWorkers[i].setType(TopicModel_worker.RunType.RT_inference);
            m_topicWorkers[i].clearObjects();
        }

        for(int i = 0; i < m_userWorkers.length; i++){
            m_userWorkers[i].setType(TopicModel_worker.RunType.RT_inference);
            m_userWorkers[i].clearObjects();
        }

        //evenly allocate the testing work load
        int workerID = 0;
        for (_Doc d : m_testSet) {
            m_workers[workerID % m_workers.length].addDoc(d);
            workerID++;
        }

        workerID = 0;
        for(_Topic4EUB t: m_topics){
            m_topicWorkers[workerID % m_topicWorkers.length].addObject(t);
            workerID++;
        }

        workerID = 0;
        for(_User4EUB u: m_testUserSet){
            m_userWorkers[workerID % m_userWorkers.length].addObject(u);
            workerID++;
        }

        do {
            init();
            perplexity = 0.0;
            totalWords = 0;

            // doc
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

            for(int i = 0; i < m_topicWorkers.length; i++){
                m_threadpool[i] = new Thread(m_topicWorkers[i]);
                m_threadpool[i].start();
            }
            for (Thread thread : m_threadpool) {
                try {
                    thread.join();
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            }

            for(int i=0; i < m_userWorkers.length; i++){
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

            for (TopicModelWorker worker : m_workers) {
                perplexity += ((Doc_worker) worker).getPerplexity();
                totalWords += ((Doc_worker) worker).getTotalWords();
            }

            if(Double.isNaN(perplexity) || Double.isInfinite(perplexity)){
                System.err.format("[Error]Inference generate NaN\n");
                break;
            }

            if(iter > 0)
                converge = Math.abs((perplexity - last) / last);
            else
                converge = 1.0;

            last = perplexity;
            System.out.format("[Inference]Likelihood: %.2f\n", last);
            if(converge < m_varConverge)
                break;
        }while(iter++ < m_varMaxIter);

        return perplexity;
    }

    public void printPerplexity(double[] performance){
        String[] groups = new String[]{"light", "medium", "heavy"};
        System.out.println();
        for(int i=0; i<groups.length; i++) {
            System.out.format("%s:\t%.4f\n", groups[i], performance[i]);
        }
    }
}
