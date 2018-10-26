package myMains;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;

import opennlp.tools.util.InvalidFormatException;
import structures._User;
import Analyzer.MultiThreadedLMAnalyzer;
import Analyzer.MultiThreadedReviewAnalyzer;
import Application.CollaborativeFiltering.CollaborativeFiltering;
import Application.CollaborativeFiltering.CollaborativeFilteringWithETBIR;
import Application.CollaborativeFiltering.CollaborativeFilteringWithMMB;

public class MyETBIRCFMain {
	public static void main(String[] args) throws InvalidFormatException, FileNotFoundException, IOException{

		int classNumber = 5;
		int Ngram = 2; // The default value is unigram.
		int lengthThreshold = 5; // Document length threshold
		double trainRatio = 0, adaptRatio = 1;
		int numberOfCores = Runtime.getRuntime().availableProcessors();
		boolean enforceAdapt = true;

		String dataset = "yelp"; // "Amazon", "Yelp"
		String tokenModel = "./data/Model/en-token.bin"; // Token model.
		
		String fs = "DF";//"IG_CHI"
		int lmTopK = 1000; // topK for language model.
		String lmFvFile = String.format("./data/CoLinAdapt/%s/fv_lm_%s_%d.txt", dataset, fs, lmTopK);
		
		String providedCV = String.format("./data/ETBIR/%s/%s_features.txt", dataset, dataset); // CV.
		String trainFolder = String.format("./data/ETBIR/%s/train", dataset);
		String testFolder = String.format("./data/ETBIR/%s/test", dataset);

//		String providedCV = String.format("/zf8/lg5bt/DataSigir/%s/SelectedVocab.csv", dataset); // CV.
//		String userFolder = String.format("/zf8/lg5bt/DataSigir/%s/Users", dataset);
				
		MultiThreadedLMAnalyzer analyzer = new MultiThreadedLMAnalyzer(tokenModel, classNumber, providedCV, lmFvFile, Ngram, lengthThreshold, numberOfCores, true);
		analyzer.config(trainRatio, adaptRatio, enforceAdapt);
		analyzer.loadUserDir(trainFolder); // load user and reviews
		analyzer.loadTestUserDir(testFolder);
		analyzer.setFeatureValues("TFIDF-sublinear", 0);
//		analyzer.rmMultipleReviews4OneItem();
		
		/***Collaborative filtering starts here.***/
		boolean equalWeight = false;
		String dir, model, cfFile;
		String suffix1 = "txt", suffix2 = "classifer";
//		String[] models = new String[]{"etbir"};
//		String[] models = new String[]{"mtsvm_0.5_1", "mtclindp_0.5_1", "mtclinhdp_0.5", "mmb_lm", "mmb_lr", "mmb_mixture"};
		String neighborSelection = "all"; // "all" 
		
		/***
		 * In order to perform cf, we need to follow the following steps:
		 * Step 1: construct ranking neighbors using the same CollaborativeFiltering.java
		 * Step 2: perform collaborative filtering */
		
		// Step 1: construct ranking neighbors using the same CollaborativeFiltering.java
		CollaborativeFiltering cfInit = new CollaborativeFiltering(analyzer.getUsers(), analyzer.getFeatureSize()+1);
		int dim = 20;
		int[] threshold = new int[]{5};
		
		for(int th: threshold){	
			dir = String.format("./data/cfData/%s_cf_%s_%d_", dataset, neighborSelection, th);
			// construct ranking neighbors
			cfInit.constructRankingNeighbors(neighborSelection, th);
			cfInit.saveUserItemPairs(dir);
		}
		
		// Step 2: perform collaborative filtering
		int[] ks = new int[]{4, 6, 8, 10}; // top_k neighbors
		for(int th: threshold){ // threshold: time or popularity
			for(int k: ks){
				// load the saved neighbor file
				cfFile = String.format("./data/cfData/%s_cf_%s_%d_test.csv", dataset, neighborSelection, th);
				ArrayList<_User> cfUsers = cfInit.getUsers();
				cfInit.loadRankingCandidates(cfFile);
			
				int validUser = cfInit.getValidUserSize();
				System.out.format("\n-----------------run ETBIR %d neighbors-------------------------\n", k);
				CollaborativeFilteringWithETBIR cf = new CollaborativeFilteringWithETBIR(cfUsers, analyzer.getFeatureSize()+1, k, dim);
				cf.setValidUserSize(validUser);
				cf.setEqualWeightFlag(equalWeight);
				
				String userWeight = "./data/ETBIR/yelp/output/ETBIR_final_p4User.txt";
				String itemWeight = "./data/ETBIR/yelp/output/ETBIR_final_eta4Item.txt";
				cf.loadWeights(userWeight, "", suffix1, suffix2);
				cf.loadItemWeights(itemWeight);
				
				cf.calculateAllNDCGMAP();
				cf.calculateAvgNDCGMAP();

				System.out.format("\n[Info] NDCG: %.4f, MAP: %.4f\n", cf.getAvgNDCG(), cf.getAvgMAP());
			}
		}
	}
}
