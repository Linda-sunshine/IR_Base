package mains;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;

import opennlp.tools.util.InvalidFormatException;
import structures._CFUser;
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
				
		/***Collaborative filtering starts here.***/
		boolean neiAll = true;
		String dir, model;
		String suffix1 = "txt", suffix2 = "classifer";
		String[] models = new String[]{"mtclinmmb_0.5"};
		
		if(!neiAll){
			int t = 2, k = 4;
			CollaborativeFiltering cfInit = new CollaborativeFiltering(analyzer.getUsers(), analyzer.getFeatureSize()+1, k, t);
			// construct ranking neighbors
			cfInit.constructRankingNeighbors();
			ArrayList<_CFUser> cfUsers = cfInit.getUsers();
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
			
			String filename = String.format("./data/%s_cf_%d_top%d.txt", dataset, t, k);
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
			
			CollaborativeFilteringWithAllNeighbors cfInit = new CollaborativeFilteringWithAllNeighbors(analyzer.getUsers());
			// construct ranking neighbors
			cfInit.constructRankingNeighbors();
			ArrayList<_CFUser> cfUsers = cfInit.getUsers();
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
				// utilize the average as ranking score
				if(model.equals("avg"))
					cf.setAvgFlag(true);
				else
					cf.loadWeights(dir, suffix1, suffix2);
				
				cf.calculateAllNDCGMAP();
				cf.calculateAvgNDCGMAP();
				performance[m][0] = cf.getAvgNDCG();
				performance[m][1] = cf.getAvgMAP();
			}
			String filename = String.format("./data/%s_cf_all_nei.txt", dataset);
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
