package mains;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.regex.Pattern;

import opennlp.tools.util.InvalidFormatException;
import structures.CFParameter;
import structures._User;
import Analyzer.MultiThreadedLMAnalyzer;
import Application.CollaborativeFiltering;
import Application.CollaborativeFilteringWithAllNeighbors;
import Application.CollaborativeFilteringWithItem;
import Application.CollaborativeFilteringWithMMB;

public class CFExecution {
	public static void main(String[] args) throws InvalidFormatException, FileNotFoundException, IOException{

		int classNumber = 5;
		int Ngram = 2; // The default value is unigram.
		int lengthThreshold = 5; // Document length threshold
		double trainRatio = 0, adaptRatio = 0.5;
		int numberOfCores = Runtime.getRuntime().availableProcessors();
		boolean enforceAdapt = true;
		CFParameter param = new CFParameter(args);
		
		String tokenModel = "./data/Model/en-token.bin"; // Token model.
		
//		String providedCV = String.format("./data/CoLinAdapt/%s/SelectedVocab.csv", param.m_data); // CV.
//		String userFolder = String.format("./data/CoLinAdapt/%s/Users", param.m_data);
		
		String providedCV = String.format("/zf8/lg5bt/DataSigir/%s/SelectedVocab.csv", param.m_data); // CV.
		String userFolder = String.format("/zf8/lg5bt/DataSigir/%s/Users", param.m_data);

		String fs = "DF";//"IG_CHI"
		int lmTopK = 1000; // topK for language model.
		String lmFvFile = String.format("/zf8/lg5bt/DataSigir/%s/fv_lm_%s_%d.txt", param.m_data, fs, lmTopK);
		
		MultiThreadedLMAnalyzer analyzer = new MultiThreadedLMAnalyzer(tokenModel, classNumber, providedCV, lmFvFile, Ngram, lengthThreshold, numberOfCores, false);
		analyzer.config(trainRatio, adaptRatio, enforceAdapt);
		analyzer.loadUserDir(userFolder); // load user and reviews
		analyzer.setFeatureValues("TFIDF-sublinear", 0);			
		analyzer.rmMultipleReviews4OneItem();

		/***Collaborative filtering starts here.***/
		String dir, model;
		String suffix1 = "txt", suffix2 = "classifer";
		String[] models = new String[]{"avg", "lr", "lm", "mtsvm_0.5_1", "mtclindp_0.5_1", "mtclinhdp_0.5", "mmb_lm", "mmb_lr", "mmb_mixture"};
		long start = System.currentTimeMillis();

		// if we select time*review_size as candidate reviews 
		if(!param.m_neiAll){
			CollaborativeFiltering cfInit = new CollaborativeFiltering(analyzer.getUsers(), analyzer.getFeatureSize()+1, param.m_k, param.m_t);
//			cfInit.constructRankingNeighbors();
			String cfFile = String.format("/zf8/lg5bt/DataSigir/%s/cfData/%s_cf_time_%d_topk_%d_test.csv", param.m_data, param.m_data, param.m_t, param.m_k);

			ArrayList<_User> cfUsers = cfInit.getUsers();
			cfInit.loadRankingCandidates(cfFile);

			int validUser = cfInit.getValidUserSize();
			double[][] performance = new double[models.length][2];
			
			for(int m=0; m<models.length; m++){
				model = models[m];
				dir = String.format("/zf8/lg5bt/DataSigir/%s/models/%s_%s/", param.m_data, param.m_data, model);
				System.out.format("\n-----------------run %s %d neighbors-------------------------\n", model, param.m_k);
				
				CollaborativeFiltering cf = null;
				if(Pattern.matches("mmb_mixture.*", model)){
					cf = new CollaborativeFilteringWithMMB(cfUsers, analyzer.getFeatureSize()+1, param.m_k);
					((CollaborativeFilteringWithMMB) cf).calculateMLEB(dir+"B_0.txt", dir+"B_1.txt");
				} else 
					cf = new CollaborativeFiltering(cfUsers, analyzer.getFeatureSize()+1, param.m_k);
				
				cf.setEqualWeightFlag(param.m_equalWeight);
				cf.setValidUserSize(validUser);
				
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
				cf.savePerf(String.format("%s_perf_%s_equalWeight_%b_time_%d_top_%d.txt", param.m_data, model, param.m_equalWeight, param.m_t, param.m_k));
				
				performance[m][0] = cf.getAvgNDCG();
				performance[m][1] = cf.getAvgMAP();
				System.err.format("\n----------------finish running %s with topk neighbors-------------------------\n", model);
			}
			
			String filename = String.format("./data/%s_cf_equalWeight_%b_time_%d_top_%d.txt", param.m_data, param.m_equalWeight, param.m_t, param.m_k);
			PrintWriter writer = new PrintWriter(new File(filename));
			writer.write("\t\tNDCG\tMAP\n");

			for(int m=0; m<models.length; m++){
				writer.write(models[m]+"\t");
				for(double p: performance[m])
					writer.write(p+"\t");
				writer.write("\n");
			}
			writer.close();
			
		} else{
			CollaborativeFilteringWithAllNeighbors cfInit = new CollaborativeFilteringWithAllNeighbors(analyzer.getUsers(), analyzer.getFeatureSize(), param.m_pop);
//			cfInit.constructRankingNeighbors();
			String cfFile = String.format("/zf8/lg5bt/DataSigir/%s/cfData/%s_cf_all_nei_pop_%d_test.csv", param.m_data, param.m_data, param.m_pop);
			
			ArrayList<_User> cfUsers = cfInit.getUsers();
			cfInit.loadRankingCandidates(cfFile);
			
			int validUser = cfInit.getValidUserSize();
			double[][] performance = new double[models.length][2];
			
			for(int m=0; m<models.length; m++){
				model = models[m];
				dir = String.format("/zf8/lg5bt/DataSigir/%s/models/%s_%s/", param.m_data, param.m_data, model);
				System.out.format("\n-----------------run %s with all neighbors pop: %d-------------------------\n", model, param.m_pop);
			
				CollaborativeFiltering cf = null;
				if(Pattern.matches("mmb_mixture.*", model)){
					cf = new CollaborativeFilteringWithMMB(cfUsers, analyzer.getFeatureSize()+1, param.m_k);
					((CollaborativeFilteringWithMMB) cf).calculateMLEB(dir+"B_0.txt", dir+"B_1.txt");
				} else if(model.equals("item_lm") || model.equals("item_lr")){
					cf = new CollaborativeFilteringWithItem(cfUsers, analyzer.getFeatureSize()+1);
				} else
					cf = new CollaborativeFiltering(cfUsers, analyzer.getFeatureSize()+1, param.m_k);
				
				cf.setEqualWeightFlag(param.m_equalWeight);
				cf.setValidUserSize(validUser);
				
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
				cf.savePerf(String.format("%s_perf_%s_equalWeight_%b_topk_%d_pop_%d_all_nei.txt", param.m_data, model, param.m_equalWeight, param.m_k, param.m_pop));

				performance[m][0] = cf.getAvgNDCG();
				performance[m][1] = cf.getAvgMAP();
				System.out.format("\n----------------finish running %s with all neighbors pop %d-------------------------\n", model, param.m_pop);
			}
			String filename = String.format("./data/%s_cf_equalWeight_%b_topk_%d_pop_%d_all_nei.txt", param.m_data, param.m_equalWeight, param.m_k, param.m_pop);
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
		
		long end = System.currentTimeMillis();
		long mins = (end - start)/(1000*60);
		System.out.println("[Info]The collaborative filtering took " + mins + " mins to finish!");
	}
}
