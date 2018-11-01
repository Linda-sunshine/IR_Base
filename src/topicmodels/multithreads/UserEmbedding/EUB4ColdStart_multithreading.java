package topicmodels.multithreads.UserEmbedding;

import structures._Corpus;
import structures._Doc;
import structures._Review;
import topicmodels.multithreads.TopicModelWorker;

import java.util.ArrayList;

/**
 * @author Lin Gong (lg5bt@virginia.edu)
 */
public class EUB4ColdStart_multithreading extends EUB_multithreading {

    ArrayList<_Doc> m_testSetLight;
    ArrayList<_Doc> m_testSetMedium;
    ArrayList<_Doc> m_testSetHeavy;
    ArrayList<ArrayList<_Doc>> m_allTestSets;

    public EUB4ColdStart_multithreading(int number_of_iteration, double converge, double beta, _Corpus c, double lambda,
                                        int number_of_topics, double alpha, int varMaxIter, double varConverge, int m) {
        super(number_of_iteration, converge, beta, c, lambda, number_of_topics, alpha, varMaxIter, varConverge, m);
        m_testSetLight = new ArrayList<>();
        m_testSetMedium = new ArrayList<>();
        m_testSetHeavy = new ArrayList<>();
        m_allTestSets = new ArrayList<>();
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

        m_allTestSets.clear();
        m_testSetLight.clear();
        m_testSetMedium.clear();
        m_testSetHeavy.clear();

        for(int i=0; i<m_users.size(); i++){
            for(_Review r: m_users.get(i).getReviews()){
                if(r.getMask4CV() == 0){
                    r.setType(_Doc.rType.TEST);
                    m_testSetLight.add(r);
                } else if(r.getMask4CV() == 1) {
                    r.setType(_Doc.rType.TEST);
                    m_testSetMedium.add(r);
                } else if(r.getMask4CV() == 2){
                    r.setType(_Doc.rType.TEST);
                    m_testSetHeavy.add(r);
                } else{
                    r.setType(_Doc.rType.TRAIN);
                    m_trainSet.add(r);
                }
            }
        }
        buildUserDocMap();
        EM();
        System.out.format("In one fold, (train: test_light: test_medium : test_heavy)=(%d : %d : %d : %d)\n",
                m_trainSet.size(), m_testSetLight.size(), m_testSetMedium.size(), m_testSetHeavy.size());
        if(m_mType == modelType.CV4DOC){
            System.out.println("[Info]Current mode is cv for docs, start evaluation....");
            int[] inferIters = new int[]{500, 1000};
            m_allTestSets.add(m_testSetLight);
            m_allTestSets.add(m_testSetMedium);
            m_allTestSets.add(m_testSetHeavy);
            double[][] allPerplexity = new double[m_allTestSets.size()][inferIters.length];
            for(int i=0; i<m_allTestSets.size(); i++) {
                for (int j=0; j < inferIters.length; j++) {
                    m_testSet = m_allTestSets.get(i);
                    perplexity = evaluation(inferIters[j]);
                    allPerplexity[i][j] = perplexity;
                }
            }
            System.out.println("================Perplexity===============");
            printPerplexity(inferIters, allPerplexity);
        } else if(m_mType == modelType.CV4EDGE){
            System.out.println("[Info]Current mode is cv for edges, link predication is performed later.");
        } else{
            System.out.println("[error]Please specify the correct mode for evaluation!");
        }
        printStat4OneFold(k, saveDir, perplexity);
    }

    public void printPerplexity(int[] inferIters, double[][] performance){
        System.out.format("Infer iter:\t");
        for(int iter: inferIters)
            System.out.format("%d\t", iter);
        String[] groups = new String[]{"light", "medium", "heavy"};
        for(int i=0; i<groups.length; i++) {
            System.out.format("\n%s:\t", groups[i]);
            for (double p : performance[i]) {
                System.out.format("%.4f\t", p);
            }
        }
        System.out.println();
    }
}
