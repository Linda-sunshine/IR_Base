package mains;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashMap;

import opennlp.tools.util.InvalidFormatException;

import Analyzer.CFMultiThreadedUserAnalyzer;
import Analyzer.MultiThreadedUserAnalyzer;
import Application.CollaborativeFiltering;

public class CFMain {
	public static void main(String[] args) throws InvalidFormatException, FileNotFoundException, IOException{

		int classNumber = 5;
		int Ngram = 2; // The default value is unigram.
		int lengthThreshold = 5; // Document length threshold
		double trainRatio = 0, adaptRatio = 0.5;
		int numberOfCores = Runtime.getRuntime().availableProcessors();
		boolean enforceAdapt = true;

		String dataset = "Amazon"; // "Amazon", "Yelp"
		String tokenModel = "./data/Model/en-token.bin"; // Token model.
		
//		String providedCV = String.format("./data/CoLinAdapt/%s/SelectedVocab.csv", dataset); // CV.
//		String userFolder = String.format("./data/CoLinAdapt/%s/Users", dataset);
		
		String providedCV = String.format("/if15/lg5bt/DataSigir/%s/SelectedVocab.csv", dataset); // CV.
		String userFolder = String.format("/if15/lg5bt/DataSigir/%s/Users", dataset);

		CFMultiThreadedUserAnalyzer analyzer = new CFMultiThreadedUserAnalyzer(tokenModel, classNumber, providedCV, Ngram, lengthThreshold, numberOfCores);
		analyzer.config(trainRatio, adaptRatio, enforceAdapt);
		analyzer.loadUserDir(userFolder); // load user and reviews
		analyzer.setFeatureValues("TFIDF-sublinear", 0);	
				
		/***Collaborative filtering starts here.***/
		for(int t=2; t<11; t+=2){
		// Construct the rdmNeighbors first.
		String model, dir;
		int topK = 0;
		CollaborativeFiltering cfInit = new CollaborativeFiltering(analyzer.getUsers(), t);
		cfInit.init();
		HashMap<String, ArrayList<Integer>> userIDRdmNeighbors = cfInit.constructRandomNeighbors();

		String[] models = new String[]{"mtclindp_u_0.5", "mtsvm_0.5", "clindp_0.5", "clinkmeans_0.5", "clrdp_0.5", "mtclrdp_0.5", "linadapt_0.5"};
		int[] ks = new int[]{2, 4, 6};
		String suffix = "classifer";
		double[][] performance = new double[models.length][ks.length*2];
		for(int m=0; m<models.length; m++){
			model = models[m];
//			dir = String.format("/home/lin/DataWsdm2017/%s/CF/%s_%s", dataset, dataset, model);
			dir = String.format("/if15/lg5bt/DataWsdm2017/%s/CF/%s_%s", dataset, dataset, model);
			for(int n=0; n < ks.length; n++){
				topK = ks[n];
				System.out.format("-----------------run %s %d neighbors-------------------------\n", model, topK);
				CollaborativeFiltering cf = new CollaborativeFiltering(analyzer.getUsers(), analyzer.getFeatureSize()+1, topK, t, model);
				cf.setUserIDRdmNeighbors(userIDRdmNeighbors);
				cf.init();
				cf.loadWeights(dir, suffix);
				cf.calculatAllNDCGMAP();
				cf.calculateAvgNDCGMAP();
				performance[m][2*n] =  cf.getAvgNDCG();
				performance[m][2*n + 1] = cf.getAvgMAP();
				double avgNuUsers = cf.getItemStat();
				System.out.println("The average number of users for each item is " + avgNuUsers);
			}
		}
//		String filename = String.format("/home/lin/DataWsdm2017/%s_cf_performance.txt", dataset);
		String filename = String.format("/if15/lg5bt/DataWsdm2017/%s/%s_cf_performance_time_%d_topK_%d.txt", dataset, dataset, t, topK);
		PrintWriter writer = new PrintWriter(new File(filename));
		
		writer.write("\t\t");
		for(int k: ks)
			writer.write("NDCG\tMAP\t");
		writer.write("\n");

		for(int m=0; m<models.length; m++){
			writer.write(models[m]+"\t");
			for(double p: performance[m]){
				writer.write(p+"\t");
			}
			writer.write("\n");
		}
		writer.close();
		}
	}
}
