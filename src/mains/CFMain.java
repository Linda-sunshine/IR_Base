package mains;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;

import opennlp.tools.util.InvalidFormatException;
import structures._User;
import Analyzer.MultiThreadedUserAnalyzer;
import Application.CollaborativeFiltering;
import Application.CollaborativeFilteringWithAllNeighbors;
import Application.CollaborativeFilteringWithMMB;
import Application.CollaborativeFilteringWithMMBWithAllNeighbors;

public class CFMain {
	public static void main(String[] args) throws InvalidFormatException, FileNotFoundException, IOException{

		int classNumber = 5;
		int Ngram = 2; // The default value is unigram.
		int lengthThreshold = 5; // Document length threshold
		double trainRatio = 0, adaptRatio = 0.5;
		int numberOfCores = Runtime.getRuntime().availableProcessors();
		boolean enforceAdapt = true;

		String dataset = "YelpNew"; // "Amazon", "Yelp"
		String tokenModel = "./data/Model/en-token.bin"; // Token model.
		
		String providedCV = String.format("./data/CoLinAdapt/%s/SelectedVocab.csv", dataset); // CV.
		String userFolder = String.format("./data/CoLinAdapt/%s/Users", dataset);
		
//		String providedCV = String.format("/zf8/lg5bt/DataSigir/%s/SelectedVocab.csv", dataset); // CV.
//		String userFolder = String.format("/zf8/lg5bt/DataSigir/%s/Users", dataset);

		MultiThreadedUserAnalyzer analyzer = new MultiThreadedUserAnalyzer(tokenModel, classNumber, providedCV, Ngram, lengthThreshold, numberOfCores, false);
		analyzer.config(trainRatio, adaptRatio, enforceAdapt);
		analyzer.loadUserDir(userFolder); // load user and reviews
		analyzer.setFeatureValues("TFIDF-sublinear", 0);	
		analyzer.rmMultipleReviews4OneItem();
		
		/***Collaborative filtering starts here.***/
		boolean neiAll = false;
		boolean equalWeight = false;
		String dir, model;
		String suffix1 = "txt", suffix2 = "classifer";
//		String[] models = new String[]{"fm"};
		String[] models = new String[]{"avg", "mtsvm_0.5_1", "mtclindp_0.5_1", "mtclinhdp_0.5", "mtclinmmb_0.5_old", "mtclinmmb_0.5_new", "mmb_mixture"};
//		String[] models = new String[]{"avg", "mmb_mixture"};

		if(!neiAll){
			for(int t: new int[]{2,3,4,5}){
				for(int k: new int[]{4,6,8,10}){
			
			CollaborativeFiltering cfInit = new CollaborativeFiltering(analyzer.getUsers(), analyzer.getFeatureSize()+1, k, t);
			// construct ranking neighbors
//			cfInit.constructRankingNeighbors();
			String cfFile = String.format("./data/cfData/%s_cf_time_%d_topk_%d_test.csv", dataset, t, k);
//			cfInit.saveUserItemPairs(cfFile);
			
			ArrayList<_User> cfUsers = cfInit.getUsers();
			cfInit.loadRankingCandidates(cfFile);
			int validUser = cfInit.getValidUserSize();
			double[][] performance = new double[models.length][2];
			
			for(int m=0; m<models.length; m++){
				model = models[m];
				dir = String.format("/home/lin/DataSigir/%s/models/%s_%s/", dataset, dataset, model);
				System.out.format("\n-----------------run %s %d neighbors-------------------------\n", model, k);
				
				CollaborativeFiltering cf = null;
				if(model.equals("mmb_mixture")){
					cf = new CollaborativeFilteringWithMMB(cfUsers, analyzer.getFeatureSize()+1, k);
					((CollaborativeFilteringWithMMB) cf).calculateMLEB(dir+"B_0.txt", dir+"B_1.txt");
				} else 
					cf = new CollaborativeFiltering(cfUsers, analyzer.getFeatureSize()+1, k);
				cf.setValidUserSize(validUser);
				cf.setEqualWeightFlag(equalWeight);
				
				// utilize the average as ranking score
				if(model.equals("avg"))
					cf.setAvgFlag(true);
				else{
					cf.loadWeights(dir, suffix1, suffix2);
				}
				cf.calculateAllNDCGMAP();
				cf.calculateAvgNDCGMAP();
				cf.savePerf(String.format("perf_%s_equalWeight_%b_time_%d_top_%d.txt", model, equalWeight, t, k));

				performance[m][0] = cf.getAvgNDCG();
				performance[m][1] = cf.getAvgMAP();
			}
			
			String filename = String.format("./data/%s_cf_equalWeight_%b_%d_top%d.txt", dataset, equalWeight, t, k);
			PrintWriter writer = new PrintWriter(new File(filename));
			writer.write("\t\tNDCG\tMAP\n");

			for(int m=0; m<models.length; m++){
				writer.write(models[m]+"\t");
				for(double p: performance[m])
					writer.write(p+"\t");
				writer.write("\n");
			}
			writer.close();
			
		}}} else{
			
			CollaborativeFilteringWithAllNeighbors cfInit = new CollaborativeFilteringWithAllNeighbors(analyzer.getUsers());
			// construct ranking neighbors
			cfInit.constructRankingNeighbors();
			cfInit.saveUserItemPairs("./");
			
			ArrayList<_User> cfUsers = cfInit.getUsers();
			int validUser = cfInit.getValidUserSize();

			double[][] performance = new double[models.length][2];
			for(int m=0; m<models.length; m++){
				model = models[m];
				dir = String.format("/home/lin/DataSigir/%s/models/%s_%s/", dataset, dataset, model);
				System.out.format("\n-----------------run %s with all neighbors-------------------------\n", model);
			
				CollaborativeFiltering cf = null;
				if(model.equals("mmb_mixture")){
					cf = new CollaborativeFilteringWithMMBWithAllNeighbors(cfUsers, analyzer.getFeatureSize()+1);
					((CollaborativeFilteringWithMMB) cf).calculateMLEB(dir+"B_0.txt", dir+"B_1.txt");
				} else 
					cf = new CollaborativeFilteringWithAllNeighbors(cfUsers, analyzer.getFeatureSize()+1);
				
				cf.setValidUserSize(validUser);
				cf.setEqualWeightFlag(equalWeight);
				
				// utilize the average as ranking score
				if(model.equals("avg"))
					cf.setAvgFlag(true);
				else
					cf.loadWeights(dir, suffix1, suffix2);
				
				cf.calculateAllNDCGMAP();
				cf.calculateAvgNDCGMAP();
				cf.savePerf(String.format("perf_%s_equalWeight_%b_all.txt", model, equalWeight));

				performance[m][0] = cf.getAvgNDCG();
				performance[m][1] = cf.getAvgMAP();
				System.out.format("\n----------------finish running %s with all neighbors-------------------------\n", model);
			}
			String filename = String.format("./data/%s_cf_equalWeight_%b_all_nei.txt", dataset, equalWeight);
			PrintWriter writer = new PrintWriter(new File(filename));
			writer.write("\t\tNDCG\tMAP\n");

			for(int m=0; m<models.length; m++){
				writer.write(models[m]+"\t");
				for(double p: performance[m])
					writer.write(p+"\t");
				writer.write("\n");
			}
			writer.close();
		}
	}
}
