package myMains;

import Analyzer.MultiThreadedNetworkAnalyzer;
import opennlp.tools.util.InvalidFormatException;
import structures.EmbeddingParameter;
import structures._Corpus;
import topicmodels.LDA.LDA_Variational;
import topicmodels.UserEmbedding.EUB;
import topicmodels.multithreads.UserEmbedding.EUB_multithreading;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;

/**
 * @author Lin Gong (lg5bt@virginia.edu)
 */
public class MyEUBExecution {


    //In the main function, we want to input the data and do adaptation
    public static void main(String[] args) throws InvalidFormatException, FileNotFoundException, IOException {

        int classNumber = 2;
        int Ngram = 2; // The default value is unigram.
        int lengthThreshold = 5; // Document length threshold
        int numberOfCores = Runtime.getRuntime().availableProcessors();

        String tokenModel = "./data/Model/en-token.bin"; // Token model.

        EmbeddingParameter param = new EmbeddingParameter(args);
        String providedCV = String.format("%s/%s/SelectedVocab.csv", param.m_prefix, param.m_data);
        String userFolder = String.format("%s/%s/Users", param.m_prefix, param.m_data);
        String friendFile = String.format("%s/%s/%sFriends.txt", param.m_prefix, param.m_data, param.m_data);
        String cvIndexFile = String.format("%s/%s/%sCVIndex.txt", param.m_prefix, param.m_data, param.m_data);

        int kFold = 5;
        MultiThreadedNetworkAnalyzer analyzer = new MultiThreadedNetworkAnalyzer(tokenModel, classNumber, providedCV,
                Ngram, lengthThreshold, numberOfCores, false);
        analyzer.setAllocateReviewFlag(false); // do not allocate reviews

        // we store the interaction information before-hand, load them directly
        analyzer.loadUserDir(userFolder);
        analyzer.constructUserIDIndex();
        analyzer.loadInteractions(friendFile);
        analyzer.loadCVIndex(cvIndexFile);

        /***Start running joint modeling of user embedding and topic embedding****/
        double emConverge = 1e-10, alpha = 1 + 1e-2, beta = 1 + 1e-3, lambda = 1 + 1e-3, varConverge = 1e-6;//these two parameters must be larger than 1!!!
        _Corpus corpus = analyzer.getCorpus();

        long start = System.currentTimeMillis();
        LDA_Variational tModel = null;
        if(param.m_multiFlag) {
            tModel = new EUB_multithreading(param.m_emIter, emConverge, beta, corpus, lambda, param.m_number_of_topics,
                    alpha, param.m_varIter, varConverge, param.m_embeddingDim);
        } else {
            tModel = new EUB(param.m_emIter, emConverge, beta, corpus, lambda, param.m_number_of_topics, alpha,
                    param.m_varIter, varConverge, param.m_embeddingDim);
        }
        ((EUB) tModel).setModelParamsUpdateFlags(param.m_alphaFlag, param.m_gammaFlag, param.m_betaFlag,
                param.m_tauFlag, param.m_xiFlag);
        ((EUB) tModel).initLookupTables(analyzer.getUsers());
        ((EUB) tModel).setDisplayLv(0);
        ((EUB) tModel).setStepSize(param.m_stepSize);

        String savePrefix = String.format("%s/%s_%d", param.m_savePrefix, param.m_data, start);
        ((EUB) tModel).fixedCrossValidation(kFold, savePrefix);
        long end = System.currentTimeMillis();

        System.out.println("\n[Info]Start time: " + start);
        // the total time of training and testing in the unit of hours
        double hours = (end - start)/(1000*60*60);
        System.out.print(String.format("[Time]This training+testing process took %.2f hours.\n", hours));

    }
}
