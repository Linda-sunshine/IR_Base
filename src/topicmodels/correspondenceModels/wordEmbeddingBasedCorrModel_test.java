package topicmodels.correspondenceModels;

import structures.*;
import utils.Utils;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Date;
import java.util.TreeMap;

/**
 * Created by jetcai1900 on 3/19/17.
 */
public class wordEmbeddingBasedCorrModel_test extends wordEmbeddingBasedCorrModel{
    public wordEmbeddingBasedCorrModel_test(int number_of_iteration, double converge, double beta, _Corpus c,
                                            double lambda, int number_of_topics, double alpha, double alpha_c,
                                            double[]gamma, double burnIn, int lag){
        super(number_of_iteration, converge, beta, c,
        lambda, number_of_topics, alpha, alpha_c,
        gamma, burnIn, lag);

    }

    public void printTopWords(int k, String betaFile) {

        double loglikelihood = calculate_log_likelihood();
        System.out.format("Final Log Likelihood %.3f\t", loglikelihood);

        String filePrefix = betaFile.replace("topWords.txt", "");
        debugOutput(filePrefix);

        Arrays.fill(m_sstat, 0);

        System.out.println("print top words");
        for (_Doc d : m_trainSet) {
            for (int i = 0; i < number_of_topics; i++)
                m_sstat[i] += m_logSpace ? Math.exp(d.m_topics[i])
                        : d.m_topics[i];
        }

        Utils.L1Normalization(m_sstat);

        try {
            System.out.println("beta file");
            PrintWriter betaOut = new PrintWriter(new File(betaFile));
            for (int i = 0; i < topic_term_probabilty.length; i++) {
                MyPriorityQueue<_RankItem> fVector = new MyPriorityQueue<_RankItem>(
                        k);
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

    public void debugOutput(String filePrefix) {

        File topicFolder = new File(filePrefix + "topicAssignment");

        if (!topicFolder.exists()) {
            System.out.println("creating directory" + topicFolder);
            topicFolder.mkdir();
        }

        File childTopKStnFolder = new File(filePrefix + "topKStn");
        if (!childTopKStnFolder.exists()) {
            System.out.println("creating top K stn directory\t"
                    + childTopKStnFolder);
            childTopKStnFolder.mkdir();
        }

        File stnTopKChildFolder = new File(filePrefix + "topKChild");
        if (!stnTopKChildFolder.exists()) {
            System.out.println("creating top K child directory\t"
                    + stnTopKChildFolder);
            stnTopKChildFolder.mkdir();
        }

        int topKStn = 10;
        int topKChild = 10;

        File parentChildTopicDistributionFolder = new File(filePrefix
                + "topicProportion");
        if (!parentChildTopicDistributionFolder.exists()) {
            System.out.println("creating topic distribution folder\t"
                    + parentChildTopicDistributionFolder);
            parentChildTopicDistributionFolder.mkdir();
        }

        File xFolder = new File(filePrefix+"xAssignment");
        if(!xFolder.exists()){
            System.out.println("creating directory for x assignment"+xFolder);
            xFolder.mkdir();
        }

        for (_Doc d : m_trainSet) {
            if (d instanceof _ParentDoc) {
                printParentTopicAssignment(d, topicFolder);
//                printParameter(d, parentChildTopicDistributionFolder);

            } else if (d instanceof _ChildDoc) {
                printChildTopicAssignment(d, topicFolder);
                printXAssignment(d, xFolder);
            }
            // if(d instanceof _ParentDoc){
            // printTopKChild4Stn(topKChild, (_ParentDoc)d, stnTopKChildFolder);
            // printTopKStn4Child(topKStn, (_ParentDoc)d, childTopKStnFolder);
            // }
        }

        String parentParameterFile = filePrefix + "parentParameter.txt";
        String childParameterFile = filePrefix + "childParameter.txt";

        printParameter(parentParameterFile, childParameterFile, m_trainSet);
        // printTestParameter4Spam(filePrefix);

//        String similarityFile = filePrefix + "topicSimilarity.txt";
//        discoverSpecificComments(similarityFile);
//        printEntropy(filePrefix);
        printTopKChild4Parent(filePrefix, topKChild);
        printXVal(filePrefix);

//        printTopKChild4Stn(filePrefix, topKChild);
//        printTopKChild4StnWithHybrid(filePrefix, topKChild);
//        printTopKChild4StnWithHybridPro(filePrefix, topKChild);
//        printTopKStn4Child(filePrefix, topKStn);
//        printTopicWordDistribution(filePrefix);
//        int randomNum = 5;
//        int topK = 5;
//        selectStn(filePrefix, topK, randomNum);

    }

    protected void printParentTopicAssignment(_Doc d, File topicFolder) {
        _ParentDoc pDoc = (_ParentDoc) d;
        String topicAssignmentFile = pDoc.getName() + ".txt";
        try {
            PrintWriter pw = new PrintWriter(new File(topicFolder,
                    topicAssignmentFile));

            for (_Word w : pDoc.getWords()) {
                int index = w.getIndex();
                int topic = w.getTopic();

                String featureName = m_corpus.getFeature(index);
                pw.print(featureName + ":" + topic + "\t");
            }

            pw.flush();
            pw.close();

        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
    }

    protected void printChildTopicAssignment(_Doc d, File topicFolder) {
        String topicAssignmentFile = d.getName() + ".txt";

        try {
            PrintWriter pw = new PrintWriter(new File(topicFolder,
                    topicAssignmentFile));

            for (_Word w : d.getWords()) {
                int index = w.getIndex();
                int topic = w.getTopic();

                String featureName = m_corpus.getFeature(index);
                pw.print(featureName + ":" + topic + "\t");
            }

            pw.flush();
            pw.close();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
    }

    protected void printParameter(String parentParameterFile,
                                  String childParameterFile, ArrayList<_Doc> docList) {
        System.out.println("printing parameter");
        try {
            System.out.println(parentParameterFile);
            System.out.println(childParameterFile);


            PrintWriter parentParaOut = new PrintWriter(new File(
                    parentParameterFile));
            PrintWriter childParaOut = new PrintWriter(new File(
                    childParameterFile));

            String parentMaxTopicIndexFile = parentParameterFile.replace("parentParameter.txt", "parentMaxTopicIndex.txt");
            PrintWriter parentMaxTopicPW =  new PrintWriter(new File(parentMaxTopicIndexFile));

            for (_Doc d : docList) {
                if (d instanceof _ParentDoc) {
                    parentParaOut.print(d.getName() + "\t");
                    parentParaOut.print("topicProportion\t");

                    int maximumTopicIndex = 0;
                    double maximumTopicProportion = 0;
                    for (int k = 0; k < number_of_topics; k++) {
                        parentParaOut.print(d.m_topics[k] + "\t");
                        if(maximumTopicProportion < d.m_topics[k]){
                            maximumTopicProportion = d.m_topics[k];
                            maximumTopicIndex = k;
                        }
                    }

                    parentMaxTopicPW.print(d.getName() + ":" + maximumTopicIndex+"\n");


                    for (_Stn stnObj : d.getSentences()) {
                        parentParaOut.print("sentence"
                                + (stnObj.getIndex() + 1) + "\t");
                        for (int k = 0; k < number_of_topics; k++) {
                            parentParaOut.print(stnObj.m_topics[k] + "\t");
                        }
                    }

                    parentParaOut.println();

                    for (_ChildDoc cDoc : ((_ParentDoc) d).m_childDocs) {
                        childParaOut.print(cDoc.getName() + "\t");

                        childParaOut.print("topicProportion\t");
                        for (int k = 0; k < number_of_topics; k++) {
                            childParaOut.print(cDoc.m_topics[k] + "\t");
                        }

                        childParaOut.println();
                    }
                }
            }

            parentParaOut.flush();
            parentParaOut.close();

            parentMaxTopicPW.flush();
            parentMaxTopicPW.close();

            childParaOut.flush();
            childParaOut.close();
        } catch (Exception e) {
            e.printStackTrace();
        }

    }

    protected void printTopKChild4Parent(String filePrefix, int topK) {
        String topKChild4StnFile = filePrefix + "topChild4Parent.txt";
        try {
            PrintWriter pw = new PrintWriter(new File(topKChild4StnFile));

            for (_Doc d : m_trainSet) {
                if (d instanceof _ParentDoc) {
                    _ParentDoc pDoc = (_ParentDoc) d;

                    pw.print(pDoc.getName() + "\t");

                    for (_ChildDoc cDoc : pDoc.m_childDocs) {
                        double docScore = rankChild4ParentBySim(cDoc, pDoc);

                        pw.print(cDoc.getName() + ":" + docScore + "\t");

                    }

                    pw.println();
                }
            }
            pw.flush();
            pw.close();

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    protected double rankChild4ParentBySim(_ChildDoc cDoc, _ParentDoc pDoc) {
        double childSim = Utils.cosine(cDoc.m_topics, pDoc.m_topics);

        return childSim;
    }

    protected void printXVal(String filePrefix){
        String xValFile = filePrefix+"xVal.txt";
        try{
            PrintWriter pw = new PrintWriter(new File(xValFile));
            for(_Doc d:m_trainSet){
                if (d instanceof _ParentDoc){
                    for(_ChildDoc cDoc:((_ParentDoc) d).m_childDocs )
                    pw.println(cDoc.getName()+"\t"+cDoc.m_xProportion[0]+"\t"+cDoc.m_xProportion[1]);
                }
            }
        }catch(Exception e){
            e.printStackTrace();
        }
    }

    protected void printXAssignment(_Doc d, File xFolder){
        String xFile = d.getName()+".txt";
        try{
            PrintWriter pw = new PrintWriter(new File(xFolder, xFile));
            for(_Word w:d.getWords()){
                int wId = w.getIndex();
                int xId = w.getX();

                String featureName = m_corpus.getFeature(wId);
                pw.print(featureName+":"+xId+":"+w.getXProb()+"\t");
            }

            pw.flush();
            pw.close();
        }catch(Exception e){
            e.printStackTrace();
        }
    }
}
