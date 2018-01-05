package mains;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Set;

import opennlp.tools.util.InvalidFormatException;
import structures.CFParameter;
import Analyzer.MultiThreadedUserAnalyzer;
import Application.CollaborativeFiltering;
import Application.CollaborativeFilteringWithAllNeighbors;
import Application.CollaborativeFilteringWithMMB;
import Application.CollaborativeFilteringWithMMBWithAllNeighbors;

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

		MultiThreadedUserAnalyzer analyzer = new MultiThreadedUserAnalyzer(tokenModel, classNumber, providedCV, Ngram, lengthThreshold, numberOfCores, false);
		analyzer.config(trainRatio, adaptRatio, enforceAdapt);
		analyzer.loadUserDir(userFolder); // load user and reviews
		analyzer.setFeatureValues("TFIDF-sublinear", 0);			
		
		/***Collaborative filtering starts here.***/
		// Construct the rdmNeighbors first.
		CollaborativeFiltering cfInit = new CollaborativeFiltering(analyzer.getUsers());
		HashMap<String, ArrayList<Integer>> userIDRdmNeighbors = new HashMap<String, ArrayList<Integer>>();
		
		String dir, model;
		String suffix1 = "txt", suffix2 = "classifer";
		String[] models = new String[]{"avg", "mtsvm_0.5_1", "mtclindp_0.5_1", "mtclinhdp_0.5", "mtclinmmb_0.5_old", "mtclinmm_0.5_new", "mmb_mixture"};
		long start = System.currentTimeMillis();

		// if we select time*review_size as candidate reviews 
		if(!param.m_neiAll){
			// construct random neighbors
			cfInit.constructRandomNeighbors(param.m_t, userIDRdmNeighbors);
//			String[] models = new String[]{"mmb_mixture"};
			double[][] performance = new double[models.length][2];
			for(int m=0; m<models.length; m++){
				model = models[m];
				dir = String.format("/zf8/lg5bt/DataSigir/%s/models/%s_%s/", param.m_data, param.m_data, model);
				System.out.format("\n-----------------run %s %d neighbors-------------------------\n", model, param.m_k);
				
				CollaborativeFiltering cf = null;
				if(model.equals("mmb_mixture")){
					cf = new CollaborativeFilteringWithMMB(analyzer.getUsers(), analyzer.getFeatureSize()+1, param.m_k);
					((CollaborativeFilteringWithMMB) cf).calculateMLEB(dir+"B_0.txt", dir+"B_1.txt");
				} else 
					cf = new CollaborativeFiltering(analyzer.getUsers(), analyzer.getFeatureSize()+1, param.m_k);
				// utilize the average as ranking score
				if(model.equals("avg"))
					cf.setAvgFlag(true);
				else{
					cf.loadWeights(dir, suffix1, suffix2);
				}
				
				cf.calculateAllNDCGMAP();
				cf.calculateAvgNDCGMAP();
				performance[m][0] = cf.getAvgNDCG();
				performance[m][1] = cf.getAvgMAP();
			}
			
			String filename = String.format("./data/%s_cf_%d_top%d.txt", param.m_data, param.m_t, param.m_k);
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
			cfInit.constructRandomNeighborsAll(userIDRdmNeighbors);
			
			double[][] performance = new double[models.length][2];
			for(int m=0; m<models.length; m++){
				model = models[m];
				dir = String.format("/zf8/lg5bt/DataSigir/%s/models/%s_%s/", param.m_data, param.m_data, model);
				System.out.format("\n-----------------run %s with all neighbors-------------------------\n", model);
			
				CollaborativeFiltering cf = null;
				if(model.equals("mmb_mixture")){
					cf = new CollaborativeFilteringWithMMBWithAllNeighbors(analyzer.getUsers(), analyzer.getFeatureSize()+1);
					((CollaborativeFilteringWithMMB) cf).calculateMLEB(dir+"B_0.txt", dir+"B_1.txt");
				} else 
					cf = new CollaborativeFilteringWithAllNeighbors(analyzer.getUsers(), analyzer.getFeatureSize()+1);
				// utilize the average as ranking score
				if(model.equals("avg"))
					cf.setAvgFlag(true);
				else{
					cf.loadWeights(dir, suffix1, suffix2);
				}

				cf.setUserIDRdmNeighbors(userIDRdmNeighbors);
				cf.calculateAllNDCGMAP();
				cf.calculateAvgNDCGMAP();
				performance[m][0] = cf.getAvgNDCG();
				performance[m][1] = cf.getAvgMAP();
			}
			String filename = String.format("./data/%s_cf_all_nei.txt", param.m_data);
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
