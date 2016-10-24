package mains;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashMap;
import opennlp.tools.util.InvalidFormatException;
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

		String dataset = "YelpNew"; // "Amazon", "Yelp"
		String tokenModel = "./data/Model/en-token.bin"; // Token model.
		
		String providedCV = String.format("./data/CoLinAdapt/%s/SelectedVocab.csv", dataset); // CV.
		String userFolder = String.format("./data/CoLinAdapt/%s/Users", dataset);
		
//		String providedCV = String.format("/if15/lg5bt/DataSigir/%s/SelectedVocab.csv", dataset); // CV.
//		String userFolder = String.format("/if15/lg5bt/DataSigir/%s/Users", dataset);

		MultiThreadedUserAnalyzer analyzer = new MultiThreadedUserAnalyzer(tokenModel, classNumber, providedCV, Ngram, lengthThreshold, numberOfCores, false);
		analyzer.config(trainRatio, adaptRatio, enforceAdapt);
		analyzer.loadUserDir(userFolder); // load user and reviews
		analyzer.setFeatureValues("TFIDF-sublinear", 0);	
				
		/***Collaborative filtering starts here.***/
		// Construct the rdmNeighbors first.
		String model, dir;
		int t = 2, k = 4;
		CollaborativeFiltering cfInit = new CollaborativeFiltering(analyzer.getUsers(), t);
		cfInit.init();
		HashMap<String, ArrayList<Integer>> userIDRdmNeighbors = cfInit.constructRandomNeighbors();
		
		String suffix = "classifer";
		String[] models;
		if(dataset.equals("Amazon")){
			models = new String[]{"avg", "mtclindp_0.5_1", "mtclindp_0.5_2", "mtclindp_0.5_3", "mtclindp_0.5_4",
					"clindp_0.5_1", "clindp_0.5_2", "clinkmeans_0.5_1", "clinkmeans_0.5_2", 
					"mtsvm_0.5_1", "clrdp_0.5_1", "mtclrdp_0.5_1", "linadapt_0.5_1"};
		}else{
			models = new String[]{"avg", "mtclindp_0.5_1", "mtclindp_0.5_2", "mtclindp_0.5_3",
				"clindp_0.5_1", "clinkmeans_0.5_1", "mtsvm_0.5_1", 
				"clrdp_0.5_1", "mtclrdp_0.5_1", "linadapt_0.5_1"};
		}
		
		double[][] performance = new double[models.length][2];
		for(int m=0; m<models.length; m++){
			model = models[m];
			System.out.format("\n-----------------run %s %d neighbors-------------------------\n", model, k);
			CollaborativeFiltering cf = new CollaborativeFiltering(analyzer.getUsers(), analyzer.getFeatureSize()+1, k, t, model);
			cf.setUserIDRdmNeighbors(userIDRdmNeighbors);
			
			if(model.equals("avg"))
				cf.setAvgFlag(true);
			else{
				dir = String.format("/home/lin/DataWWW2017/%s/CF/%s_%s", dataset, dataset, model);
//				dir = String.format("/if15/lg5bt/DataWWW2017/%s/CF/%s_%s", dataset, dataset, model);
				cf.loadWeights(dir, suffix);
			}
			cf.calculatAllNDCGMAP();
			String perf = String.format("perf_%s_time_%d_top_%d.txt", model, t, k);
			cf.savePerf(perf);
			cf.calculateAvgNDCGMAP();
//			System.out.println(cf.getItemStat());
			performance[m][0] = cf.getAvgNDCG();
			performance[m][1] = cf.getAvgMAP();
		}
		
		String filename = String.format("./data/%s_cf_%d_top%d.txt", dataset, t, k);
//		String filename = String.format("/if15/lg5bt/DataWWW2017/%s/%s_cf_performance_time_%d_topK_%d.txt", dataset, dataset, t, topK);
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
