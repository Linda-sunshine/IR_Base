package topicmodels.correspondenceModels;

import structures.*;
import topicmodels.multithreads.TopicModelWorker;
import utils.Utils;

import java.io.File;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;

/**
 * Created by jetcai1900 on 4/27/17.
 */
public class BitermTopicModel_test extends BitermTopicModel {
    public BitermTopicModel_test(int number_of_iteration, double converge,
                                 double beta,
                                 _Corpus c, double lambda, int number_of_topics, double alpha,
                                 double burnIn, int lag){
        super(number_of_iteration, converge, beta, c, lambda, number_of_topics,
                alpha, burnIn, lag);
    }

    public void printTopWords(int k, String betaFile) {

        double loglikelihood = calculate_log_likelihood();
        System.out.format("Final Log Likelihood %.3f\t", loglikelihood);

        String filePrefix = betaFile.replace("topWords.txt", "");
        debugOutput(filePrefix);

        System.out.println("print top words");

        try {
            System.out.println("beta file");
            PrintWriter betaOut = new PrintWriter(new File(betaFile));
            for (int i = 0; i < topic_term_probabilty.length; i++) {
                MyPriorityQueue<_RankItem> fVector = new MyPriorityQueue<_RankItem>(
                        k);
                for (int j = 0; j < vocabulary_size; j++)
                    fVector.add(new _RankItem(m_corpus.getFeature(j),
                            topic_term_probabilty[i][j]));

                betaOut.format("Topic %d(%.3f):\t", i, m_topics[i]);
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

    public void	debugOutput(String filePrefix){
        printTopicWordDistribution(filePrefix);

        String parentParameterFile = filePrefix + "parentParameter.txt";
        String childParameterFile = filePrefix + "childParameter.txt";

        printParameter(parentParameterFile, childParameterFile, m_trainSet);
    }

    protected void printTopicWordDistribution(String filePrefix) {
        String topicWordFile = filePrefix + "fullTopicWord.txt";
        try{
            PrintWriter pw = new PrintWriter(new File(topicWordFile));

            for (int k = 0; k < number_of_topics; k++) {
                pw.print(k + "\t");
                for (int v = 0; v < vocabulary_size; v++) {
                    pw.print(m_corpus.getFeature(v) + ":"
                            + topic_term_probabilty[k][v] + "\t");
                }
                pw.println();
            }

            pw.flush();
            pw.close();
        }catch (Exception e) {
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

            for (_Doc d : docList) {
                parentParaOut.print(d.getName() + "\t");
                parentParaOut.print("topicProportion\t");
                for (int k = 0; k < number_of_topics; k++) {
                    parentParaOut.print(d.m_topics[k] + "\t");
                }

                parentParaOut.println();
            }

            parentParaOut.flush();
            parentParaOut.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
