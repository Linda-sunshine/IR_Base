package topicmodels.correspondenceModels;

import structures.*;
import utils.Utils;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;

/**
 * Created by jetcai1900 on 4/29/17.
 */
public class CCTM_Online extends DCMCorrLDA {
    public CCTM_Online(int number_of_iteration, double converge, double beta,
                       _Corpus c, double lambda, int number_of_topics, double alpha_a,
                       double alpha_c, double burnIn, double ksi, double tau, int lag,
                       int newtonIter, double newtonConverge){
        super(number_of_iteration, converge, beta, c, lambda, number_of_topics, alpha_a,
         alpha_c, burnIn, ksi, tau, lag, newtonIter, newtonConverge);
    }

    public String toString(){
        return String.format("CCTM_Online[k:%d, alphaA:%.2f, beta:%.2f, trainProportion:%.2f, Gibbs Sampling]", number_of_topics, d_alpha, d_beta,
                        1 - m_testWord4PerplexityProportion);
    }

    public void LoadWordDistributions(String fileName) {
        if (fileName == null || fileName.isEmpty()) {
            return;
        }

        try{

            if (m_beta == null) {
                m_beta = new double[number_of_topics][vocabulary_size];
                m_totalBeta = new double[number_of_topics];
            }

            for (int k = 0; k < number_of_topics; k++)
                Arrays.fill(m_beta[k], 0);

            String tmpTxt;
            String[] lineContainer;
            String[] featureContainer;
            int tid = 0;

            HashMap<String, Integer> featureNameIndex = new HashMap<String, Integer>();
            for(int i=0; i<m_corpus.getFeatureSize(); i++){
                featureNameIndex.put(m_corpus.getFeature(i), featureNameIndex.size());
            }

            BufferedReader br = new BufferedReader(new InputStreamReader(
                    new FileInputStream(fileName), "UTF-8"));

            while ((tmpTxt = br.readLine()) != null) {
                tmpTxt = tmpTxt.trim();
                if (tmpTxt.isEmpty())
                    continue;

                lineContainer = tmpTxt.split("\t");

                tid = Integer.parseInt(lineContainer[0]);
                for (int i = 1; i < lineContainer.length; i++) {
                    featureContainer = lineContainer[i].split(":");

                    String featureName = featureContainer[0];
                    double featureProb = Double.parseDouble(featureContainer[1]);

                    int featureIndex = featureNameIndex.get(featureName);

                    m_beta[tid][featureIndex] = featureProb;
                }
            }

            for (int k = 0; k < number_of_topics; k++)
                m_totalBeta[k] = Utils.sumOfArray(m_beta[k]);

            System.out.println("beta is added");

        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void LoadAlphas(String fileName){
        if (fileName == null || fileName.isEmpty()) {
            return;
        }

        try{

            if (m_alpha_c == null) {
                m_alpha_c = new double[number_of_topics];
                m_alpha = new double[number_of_topics];

                m_totalAlpha = 0;
                m_totalAlpha_c = 0;
            }

            for (int k = 0; k < number_of_topics; k++) {
                Arrays.fill(m_alpha, 0);
                Arrays.fill(m_alpha_c, 0);
            }

            String tmpTxt;
            String[] lineContainer;

            BufferedReader br = new BufferedReader(new InputStreamReader(
                    new FileInputStream(fileName), "UTF-8"));

            while ((tmpTxt = br.readLine()) != null) {
                tmpTxt = tmpTxt.trim();
                if (tmpTxt.isEmpty())
                    continue;

                lineContainer = tmpTxt.split("\t");

                if(lineContainer[0].equals("alpha")) {
                    for (int i = 1; i < lineContainer.length; i++) {
                        double proportion = Double.parseDouble(lineContainer[i]);

                        m_alpha[i - 1] = proportion;
                    }
                }else{
                    for (int i = 1; i < lineContainer.length; i++) {
                        double proportion = Double.parseDouble(lineContainer[i]);

                        m_alpha_c[i - 1] = proportion;
                    }
                }
            }

            m_totalAlpha = Utils.sumOfArray(m_alpha);
            m_totalAlpha_c = Utils.sumOfArray(m_alpha_c);

            System.out.println("alpha, alpha_c is added");

        } catch (IOException e) {
            e.printStackTrace();
        }

    }

    protected void initialize_probability(Collection<_Doc> collection){
        for(_Doc d:collection){
            if(d instanceof _ParentDoc4DCM){
                _ParentDoc4DCM pDoc = (_ParentDoc4DCM)d;
                pDoc.setTopics4Gibbs(number_of_topics, 0, vocabulary_size);

                for (_Stn stnObj : d.getSentences()) {
                    stnObj.setTopicsVct(number_of_topics);
                }

                for(_ChildDoc cDoc: pDoc.m_childDocs){
                    cDoc.setTopics4Gibbs_LDA(number_of_topics, 0);
                    for(_Word w:cDoc.getWords()){
                        int wid = w.getIndex();
                        int tid = w.getTopic();

                        pDoc.m_wordTopic_stat[tid][wid] ++;
                        pDoc.m_topic_stat[tid]++;
                    }
                    computeMu4Doc(cDoc);
                }
            }

        }

    }

    public void EM(){

        m_collectCorpusStats = false;
        initialize_probability(m_trainSet);

        for (int j = 0; j < number_of_iteration; j++) {
            init();
            for (_Doc d : m_trainSet)
                calculate_E_step(d);
            calculate_M_step(j);
        }

        for(_Doc d:m_trainSet)
            estThetaInDoc(d);
    }


    public void calculate_M_step(int iter) {
        if (iter>m_burnIn && iter%m_lag == 0) {
            for (_Doc d : m_trainSet) {
                collectStats(d);
            }
        }
    }
}
