package mains;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.regex.Pattern;

import opennlp.tools.util.InvalidFormatException;
import structures._User;
import Analyzer.MultiThreadedLMAnalyzer;
import Application.CollaborativeFiltering;
import Application.CollaborativeFilteringWithAllNeighbors;
import Application.CollaborativeFilteringWithItem;
import Application.CollaborativeFilteringWithMMB;

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
		
		String fs = "DF";//"IG_CHI"
		int lmTopK = 1000; // topK for language model.
		String lmFvFile = String.format("./data/CoLinAdapt/%s/fv_lm_%s_%d.txt", dataset, fs, lmTopK);

		MultiThreadedLMAnalyzer analyzer = new MultiThreadedLMAnalyzer(tokenModel, classNumber, providedCV, lmFvFile, Ngram, lengthThreshold, numberOfCores, false);
		analyzer.config(trainRatio, adaptRatio, enforceAdapt);
		analyzer.loadUserDir(userFolder); // load user and reviews
		analyzer.setFeatureValues("TFIDF-sublinear", 0);	
		analyzer.rmMultipleReviews4OneItem();
		
		/***Collaborative filtering starts here.***/
		boolean neiAll = true;
		boolean equalWeight = false;
		String dir, model, cfFile;
		String suffix1 = "txt", suffix2 = "classifer";
//		String[] models = new String[]{"fm"};
		String[] models = new String[]{"mtsvm_0.5_1", "mtclindp_0.5_1", "mtclinhdp_0.5", "mmb_lm", "mmb_lr", "mmb_mixture"};

		if(!neiAll){
			for(int t: new int[]{5}){
				for(int k: new int[]{4,6,8,10}){
			
			dir = String.format("./data/cfData/fm/%s_cf_time_%d_topk_%d_", dataset, t, k);
			CollaborativeFiltering cfInit = new CollaborativeFiltering(analyzer.getUsers(), analyzer.getFeatureSize()+1, k, t);
			
			// construct ranking neighbors
//			cfInit.constructRankingNeighbors();
//			cfInit.saveUserItemPairs(dir);
			
			cfFile = String.format("./data/cfData/fm/%s_cf_time_%d_topk_%d_test_valid.csv", dataset, t, k);
			ArrayList<_User> cfUsers = cfInit.getUsers();
			cfInit.loadRankingCandidates(cfFile);
			
			int validUser = cfInit.getValidUserSize();
			double[][] performance = new double[models.length][2];
			
			for(int m=0; m<models.length; m++){
				model = models[m];
//				dir = String.format("./data/CoLinAdapt/%s/models/%s_%s/", dataset, dataset, model);
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
				if(model.equals("avg")){
					cf.setAvgFlag(true);
				}else if(model.equals("lm") || model.equals("mmb_lm")){
					cf.setFeatureSize(lmTopK);
					cf.loadWeights(dir, model, suffix1, suffix2);
				} else{
					cf.loadWeights(dir, model, suffix1, suffix2);
				}
				
				cf.calculateAllNDCGMAP();
				cf.calculateAvgNDCGMAP();
				cf.savePerf(String.format("%s_perf_%s_equalWeight_%b_time_%d_top_%d.txt", dataset, model, equalWeight, t, k));

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
			}}
		} else{
			for(int k: new int[]{4}){
				for(int pop: new int[]{10}){

			dir = String.format("./data/cfData/fm/%s_cf_all_nei_pop_%d_", dataset, pop);
			CollaborativeFilteringWithAllNeighbors cfInit = new CollaborativeFilteringWithAllNeighbors(analyzer.getUsers(), analyzer.getFeatureSize(), pop);
			
			// construct ranking neighbors
//			cfInit.constructRankingNeighbors();
//			cfInit.saveUserItemPairs(dir);
			
			cfFile = String.format("./data/cfData/fm/%s_cf_all_nei_pop_%d_test.csv", dataset, pop);
			ArrayList<_User> cfUsers = cfInit.getUsers();
//			cfInit.calculatePopularity();
			cfInit.loadRankingCandidates(cfFile);
			
			int validUser = cfInit.getValidUserSize();
			double[][] performance = new double[models.length][2];
			
			for(int m=0; m<models.length; m++){
				model = models[m];
				dir = String.format("/home/lin/DataSigir/%s/models/%s_%s/", dataset, dataset, model);
				System.out.format("\n-----------------run %s with all neighbors pop %d-------------------------\n", model, pop);
			
				CollaborativeFiltering cf = null;
				if(Pattern.matches("mmb_mixture.*", model)){
					cf = new CollaborativeFilteringWithMMB(cfUsers, analyzer.getFeatureSize()+1, k);
					((CollaborativeFilteringWithMMB) cf).calculateMLEB(dir+"B_0.txt", dir+"B_1.txt");
				} else if(model.equals("item_lm") || model.equals("item_lr")){
					cf = new CollaborativeFilteringWithItem(cfUsers, analyzer.getFeatureSize()+1);
				} else 
					cf = new CollaborativeFiltering(cfUsers, analyzer.getFeatureSize()+1, k);
				
				cf.setValidUserSize(validUser);
				cf.setEqualWeightFlag(equalWeight);
				
				// utilize the average as ranking score
				if(model.equals("avg")){
					cf.setAvgFlag(true);
				}else if(model.equals("lm") || model.equals("mmb_lm")){
					cf.setFeatureSize(lmTopK);
					cf.loadWeights(dir, model, suffix1, suffix2); 
				} else if(model.equals("item_lm") || model.equals("item_lr")){
					String lmModel = model.split("_")[1];
					if(lmModel.equals("lm"))
						cf.setFeatureSize(lmTopK);
					cf.loadUserWeights(dir, lmModel, suffix1, suffix2);
					cf.constructItems(lmModel);
				} else{
					cf.loadWeights(dir, model, suffix1, suffix2);
				}
				
				cf.calculateAllNDCGMAP();
				cf.calculateAvgNDCGMAP();
				cf.savePerf(String.format("%s_perf_%s_equalWeight_%b_topk_%d_all.txt", dataset, model, equalWeight, k));

				performance[m][0] = cf.getAvgNDCG();
				performance[m][1] = cf.getAvgMAP();
				System.out.format("\n----------------finish running %s with all neighbors pop %d-------------------------\n", model, pop);
			}
			String filename = String.format("./data/%s_cf_equalWeight_%b_topk_%d_pop_%d_all_nei.txt", dataset, equalWeight, k, pop);
			PrintWriter writer = new PrintWriter(new File(filename));
			writer.write("\t\tNDCG\tMAP\n");

			for(int m=0; m<models.length; m++){
				writer.write(models[m]+"\t");
				for(double p: performance[m])
					writer.write(p+"\t");
				writer.write("\n");
			}
			writer.close();
		}}}
	}
}
