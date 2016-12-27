package topicmodels.LDA;


import structures.*;
import utils.Utils;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;

/**
 * Created by jetcai1900 on 12/25/16.
 */
public class LDA_Variational_test extends LDA_Variational {
    public LDA_Variational_test(int number_of_iteration, double converge,
                                double beta, _Corpus c, double lambda,
                                int number_of_topics, double alpha, int varMaxIter, double varConverge){
        super(number_of_iteration, converge, beta, c, lambda, number_of_topics, alpha, varMaxIter, varConverge);
        m_logSpace = true;
    }

    public void printTopWords(int k, String betaFile) {
        double logLikelihood = calculate_log_likelihood();
        System.out.format("final log likelihood %.3f\t", logLikelihood);

        String filePrefix = betaFile.replace("topWords.txt", "");
        debugOutput(k, filePrefix);

        Arrays.fill(m_sstat, 0);

        System.out.println("print top words");
        printTopWordsDistribution(k, betaFile);
    }

    protected void debugOutput(int topK, String filePrefix) {
        File parentTopicFolder = new File(filePrefix + "parentTopicAssignment");
        File childTopicFolder = new File(filePrefix + "childTopicAssignment");

        if (!parentTopicFolder.exists()) {
            System.out.println("creating directory" + parentTopicFolder);
            parentTopicFolder.mkdir();
        }

        if (!childTopicFolder.exists()) {
            System.out.println("creating directory" + childTopicFolder);
            childTopicFolder.mkdir();
        }

        File parentWordTopicDistributionFolder = new File(filePrefix
                + "wordTopicDistribution");
        if (!parentWordTopicDistributionFolder.exists()) {
            System.out.println("creating word topic distribution folder\t"
                    + parentWordTopicDistributionFolder);
            parentWordTopicDistributionFolder.mkdir();
        }

        for (_Doc d : m_trainSet) {
            printTopicAssignment(d, parentTopicFolder);

        }

        String parameterFile = filePrefix + "parameter.txt";

        printParameter(parameterFile, m_trainSet);
    }

    protected void printTopicAssignment(_Doc d, File topicFolder) {
        String topicAssignmentFile = d.getName() + ".txt";

        try {
            PrintWriter pw = new PrintWriter(new File(topicFolder,
                    topicAssignmentFile));

            _SparseFeature[] fvs = d.getSparse();
            for(int n=0; n<fvs.length; n++){
                int wID = fvs[n].getIndex();
                String featureName = m_corpus.getFeature(wID);

                pw.print(featureName+":\n");
                for(int k=0; k<number_of_topics; k++){
                    pw.print("\t"+d.m_phi[n][k]);
                }

                pw.println();
            }

            pw.flush();
            pw.close();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
    }

    protected void printParameter(String parameterFile,
                                  ArrayList<_Doc> docList) {
        System.out.println("printing parameter");

        try {
            System.out.println(parameterFile);

            PrintWriter paraOut = new PrintWriter(new File(
                    parameterFile));

            for (_Doc d : docList) {
                int maxTopicIndex = 0;
                double maxTopicProportion = 0;
                paraOut.print(d.getName() + "\t");
                paraOut.print("topicProportion\t");
                for (int k = 0; k < number_of_topics; k++) {
                    if(m_logSpace==true)
                        paraOut.print(Math.exp(d.m_topics[k]) + "\t");
                    if(maxTopicProportion < Math.exp(d.m_topics[k])){
                        maxTopicIndex = k;
                        maxTopicProportion = Math.exp(d.m_topics[k]);
                    }
                }
                paraOut.println("maxTopicIndex\t"+maxTopicIndex);
                paraOut.println();
            }

            paraOut.flush();
            paraOut.close();

        } catch (Exception e) {
            e.printStackTrace();
        }
    }



//    protected HashMap<String, Double> rankChild4StnByLikelihood(_Stn stnObj,
//                                                                _ParentDoc4DCM pDoc) {
//        HashMap<String, Double> likelihoodMap = new HashMap<String, Double>();
//
//        for (_ChildDoc cDoc : pDoc.m_childDocs) {
//
//            double stnLogLikelihood = 0;
//            for (_Word w : stnObj.getWords()) {
//                double wordLikelihood = 0;
//                int wid = w.getIndex();
//
//                for (int k = 0; k < number_of_topics; k++) {
//                    wordLikelihood += cDoc.m_topics[k]*topic_term_probabilty[k][wid];
//                }
//
//                stnLogLikelihood += Math.log(wordLikelihood);
//
//            }
//            likelihoodMap.put(cDoc.getName(), stnLogLikelihood);
//        }
//
//        return likelihoodMap;
//    }

    protected void printTopWordsDistribution(int topK, String topWordFile) {
        Arrays.fill(m_sstat, 0);

        System.out.println("print top words");
        for (_Doc d : m_trainSet) {
            for (int i = 0; i < number_of_topics; i++)
                m_sstat[i] += m_logSpace ? Math.exp(d.m_topics[i])
                        : d.m_topics[i];
        }

        Utils.L1Normalization(m_sstat);

        try {
            System.out.println("top word file");
            PrintWriter betaOut = new PrintWriter(new File(topWordFile));
            for (int i = 0; i < topic_term_probabilty.length; i++) {
                MyPriorityQueue<_RankItem> fVector = new MyPriorityQueue<_RankItem>(
                        topK);
                for (int j = 0; j < vocabulary_size; j++)
                    fVector.add(new _RankItem(m_corpus.getFeature(j),
                            topic_term_probabilty[i][j]));

                betaOut.format("Topic %d(%.3f):\t", i, m_sstat[i]);
                for (_RankItem it : fVector) {
                    betaOut.format("%s(%.3f)\t", it.m_name,
                            m_logSpace ? Math.exp(it.m_value) : it.m_value);
                    System.out.format("%s(%.3f)\t", it.m_name,
                            m_logSpace ? Math.exp(it.m_value) : it.m_value);
                }
                betaOut.println();
                System.out.println();
            }

            betaOut.flush();
            betaOut.close();
        } catch (Exception ex) {
            System.err.print("File Not Found");
        }
    }
}
