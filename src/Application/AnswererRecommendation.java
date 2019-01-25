package Application;

import Analyzer.MultiThreadedNetworkAnalyzer;
import Application.LinkPrediction4EUB.LinkPredictionWithUserEmbedding;
import opennlp.tools.util.InvalidFormatException;
import structures._Corpus;
import topicmodels.LDA.LDA_Variational;
import topicmodels.UserEmbedding.EUB;
import topicmodels.multithreads.LDA.LDA_Variational_multithread;
import topicmodels.multithreads.UserEmbedding.EUB4ColdStart_multithreading;
import topicmodels.multithreads.UserEmbedding.EUB_multithreading;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.HashMap;

/**
 * @author Lin Gong (lg5bt@virginia.edu)
 * In this task, we would input user embedding and a set of questions' topic distributions.
 * We want to find out if the learned embedding can help us find the right person to answer the question.
 */
public class AnswererRecommendation extends LinkPredictionWithUserEmbedding {

    class _Question{
        int m_postId;
        String m_uid;


    }
    // the topic distribution for selected questions
    HashMap<Integer, double[]> m_thetas;
    HashMap<Integer, double[]> m_userEmbeddings;

    public AnswererRecommendation(){

    }

    // init recommendation
    public void initRecommendation(String idFile, String embedFile, String questionFile, String testInterFile, String testNonInterFile) {
        loadUserIds(idFile);
        loadUserEmbedding(embedFile);
//        loadQuestionIds(questionFile);

        calcSimilarity();
        // if no non-interactions are specified, we will consider all the other users as non-interactions
        // thus, non-interactions are know after loading interactions
        if(testNonInterFile == null){
            loadTestOneZeroEdges(testInterFile);
        } else{
            // otherwise, load one-edges ans zero-edges individually
            loadTestOneEdges(testInterFile);
            loadTestZeroEdges(testNonInterFile);
        }

    }


    //In the main function, we want to input the data and do adaptation
    public static void main(String[] args) {

        int classNumber = 2;
        int Ngram = 2; // The default value is unigram.
        int lengthThreshold = 5; // Document length threshold
        int numberOfCores = Runtime.getRuntime().availableProcessors();

        String dataset = "StackOverflow"; // "StackOverflow", "YelpNew"
        String tokenModel = "./data/Model/en-token.bin"; // Token model.

        String prefix = "./data/CoLinAdapt";
        String providedCV = String.format("%s/%s/%sSelectedVocab.txt", prefix, dataset, dataset);
        String userFolder = String.format("%s/%s/Users", prefix, dataset);


    }
}
