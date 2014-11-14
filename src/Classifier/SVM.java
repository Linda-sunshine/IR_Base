package Classifier;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;

import structures._Corpus;
import structures._Doc;
import structures._SparseFeature;
import utils.Utils;
import Analyzer.DocAnalyzer;
import libsvm.*;

public class SVM extends BaseClassifier{
	double m_C;
	//Constructor.
	public SVM(_Corpus c, int classNumber, int featureSize, double C){
		super(c, classNumber, featureSize);
		this.m_C = C;
	}

	//k-fold Cross Validation.
	public void crossValidation(int k, _Corpus c, int class_number){
		c.shuffle(k);
		int[] masks = c.getMasks();
		HashMap<Integer, ArrayList<_Doc>> k_folder = new HashMap<Integer, ArrayList<_Doc>>();

		// Set the hash map with documents.
		for (int i = 0; i < masks.length; i++) {
			_Doc doc = c.getCollection().get(i);
			if (k_folder.containsKey(masks[i])) {
				ArrayList<_Doc> temp = k_folder.get(masks[i]);
				temp.add(doc);
				k_folder.put(masks[i], temp);
			} else {
				ArrayList<_Doc> docs = new ArrayList<_Doc>();
				docs.add(doc);
				k_folder.put(masks[i], docs);
			}
		}
		// Set the train set and test set.
		for (int i = 0; i < k; i++) {
			this.setTestSet(k_folder.get(i));
			for (int j = 0; j < k; j++) {
				if (j != i) {
					this.addTrainSet(k_folder.get(j));
				}
			}
			// Train the data set to get the parameter.
			svm_model model = train(this.m_trainSet, this.m_C);
			test(this.m_testSet, model);
			this.m_trainSet.clear();
			// this.m_testSet.clear();
		}
		this.calculateMeanVariance(this.m_precisionsRecalls);
	}

	// Calculate the precision and recall for one folder tests.
	public double[][] calculatePreRec(double[][] tpTable) {
		double[][] PreRecOfOneFold = new double[this.m_classNo][2];
		for (int i = 0; i < this.m_classNo; i++) {
			PreRecOfOneFold[i][0] = tpTable[i][i] / sumOfColumn(tpTable, i);// Precision of the class.
			PreRecOfOneFold[i][1] = tpTable[i][i] / sumOfRow(tpTable, i);// Recall of the class.
		}
		return PreRecOfOneFold;
	}

	// Train the data set.
	public svm_model train(ArrayList<_Doc> trainSet, double C) {
		svm_model model = new svm_model();
		svm_problem problem = new svm_problem();
		problem.x = new svm_node[trainSet.size()][this.m_featureSize];
		problem.y = new double [trainSet.size()];
		svm_parameter param = new svm_parameter();
		
		//Set default value of the param.
		param = new svm_parameter();
		// default values
		param.svm_type = svm_parameter.C_SVC;
		param.kernel_type = svm_parameter.RBF;
		param.degree = 3;
		param.gamma = 0;	// 1/num_features
		param.coef0 = 0;
		param.nu = 0.5;
		param.cache_size = 100;
		param.C = C;
		param.eps = 1e-3;
		param.p = 0.1;
		param.shrinking = 1;
		param.probability = 0;
		param.nr_weight = 0;
		param.weight_label = new int[0];
		param.weight = new double[0];

		//Construct the svm_problem by enumerating all docs.
		for(int i = 0; i < trainSet.size() - 1; i++){
			_Doc temp = trainSet.get(i);
			for(int j = 0; j < temp.getSparse().length- 1; j++){
				int index = temp.getSparse()[j].getIndex();
				double value = temp.getSparse()[j].getNormValue();
				svm_node node = new svm_node();
				node.index = index;
				node.value = value;
				problem.x[i][j] = node;
			}
			problem.y[i] = temp.getYLabel();
			problem.l = trainSet.size();
		}	
		model = svm.svm_train(problem, param);
		return model;
	}

	public void test(ArrayList<_Doc> testSet, svm_model model){
		double[][] TPTable = new double [this.m_classNo][this.m_classNo];
		double[][] PreRecOfOneFold = new double[this.m_classNo][2];
		//Construct the svm_problem by enumerating all docs.
		for (int i = 0; i < testSet.size() - 1; i++) {
			_Doc temp = testSet.get(i);
			svm_node[] nodes = new svm_node[temp.getSparse().length];
			for (int j = 0; j < temp.getSparse().length - 1; j++) {
				int index = temp.getSparse()[j].getIndex();
				double value = temp.getSparse()[j].getNormValue();
				svm_node node = new svm_node();
				node.index = index;
				node.value = value;	
				nodes[j] = node;
			}
			double result = svm.svm_predict(model, nodes);
			System.out.println("The current result is " + result);
			TPTable[(int) result][temp.getYLabel()] +=1;
		}
		PreRecOfOneFold = calculatePreRec(TPTable);
		this.m_precisionsRecalls.add(PreRecOfOneFold);
	}
	
	/*****************************Main function*******************************/
	public static void main(String[] args) throws IOException{
		
		int featureSize = 0; //Initialize the fetureSize to be zero at first.
		int classNumber = 2; //Define the number of classes in this Naive Bayes.
		_Corpus corpus = new _Corpus();
		
		//The parameters used in loading files.
		String folder = "txt_sentoken";
		String suffix = ".txt";
		String tokenModel = "./data/Model/en-token.bin"; //Token model.
		String finalLocation = "/Users/lingong/Documents/Lin'sWorkSpace/IR_Base/NBFinal.txt"; //The destination of storing the final features with stats.
		String featureLocation = "/Users/lingong/Documents/Lin'sWorkSpace/IR_Base/NBSelectedFeatures.txt";
		
		//String providedCV = "";
		String featureSelection = "";
		String providedCV = "Features.txt"; //Provided CV.
		//String featureSelection = "MI"; //Feature selection method.
		
		if( providedCV.isEmpty() && featureSelection.isEmpty()){
			//Case 1: no provided CV, no feature selection.
			System.out.println("Case 1: no provided CV, no feature selection.");
			DocAnalyzer analyzer = new DocAnalyzer(tokenModel, classNumber, null, null);
			System.out.println("Start loading files, wait...");
			analyzer.LoadDirectory(folder, suffix); //Load all the documents as the data set.
			featureSize = analyzer.getFeatureSize();
			corpus = analyzer.returnCorpus(finalLocation); 
			
		} else if( !providedCV.isEmpty() && featureSelection.isEmpty()){
			//Case 2: provided CV, no feature selection.
			System.out.println("Case 2: provided CV, no feature selection.");
			System.out.println("Start loading files, wait...");
			DocAnalyzer analyzer = new DocAnalyzer(tokenModel, classNumber, providedCV, null);
			analyzer.LoadDirectory(folder, suffix); //Load all the documents as the data set.
			featureSize = analyzer.getFeatureSize();
			corpus = analyzer.returnCorpus(finalLocation); 
			
		} else if(providedCV.isEmpty() && !featureSelection.isEmpty()){
			//Case 3: no provided CV, feature selection.
			System.out.println("Case 3: no provided CV, feature selection.");
			System.out.println("Start loading files to do feature selection, wait...");
			
//			/*If the feature selection is TS, we need to load the directory three times.
//			 * 1. Load all the docs, get all the terms in the docs. Calculate the current document's similarity with other documents, find a max one??
//			 * 2. Load again to do feature selection.
//			 * 3. Load again to do classfication.
//			 * TS is not implemented yet due to the vague definition.*/
//			if(featureSelection.endsWith("TS")){
//				DocAnalyzer analyzer_1 = new DocAnalyzer(tokenModel, classNumber, null, null);
//				analyzer_1.LoadDirectory(folder, suffix); //Load all the documents as the data set.
//				analyzer_1.calculateSimilarity();
//				//analyzer_1.featureSelection(featureLocation); //Select the features.
//			}
			
			DocAnalyzer analyzer = new DocAnalyzer(tokenModel, classNumber, null, featureSelection);
			analyzer.LoadDirectory(folder, suffix); //Load all the documents as the data set.
			analyzer.featureSelection(featureLocation); //Select the features.
			
			System.out.println("Start loading files, wait...");
			DocAnalyzer analyzer_2 = new DocAnalyzer(tokenModel, classNumber, featureLocation, null);
			analyzer_2.LoadDirectory(folder, suffix);
			featureSize = analyzer.getFeatureSize();
			corpus = analyzer_2.returnCorpus(finalLocation); 
			
		} else if(!providedCV.isEmpty() && !featureSelection.isEmpty()){
			//Case 4: provided CV, feature selection.
			DocAnalyzer analyzer = new DocAnalyzer(tokenModel, classNumber, providedCV, featureSelection);
			System.out.println("Case 4: provided CV, feature selection.");
			System.out.println("Start loading file to do feature selection, wait...");
			analyzer.LoadDirectory(folder, suffix); //Load all the documents as the data set.
			analyzer.featureSelection(featureLocation); //Select the features.
			
			System.out.println("Start loading files, wait...");
			DocAnalyzer analyzer_2 = new DocAnalyzer(tokenModel, classNumber, featureLocation, null);
			analyzer_2.LoadDirectory(folder, suffix);
			featureSize = analyzer.getFeatureSize();
			corpus = analyzer_2.returnCorpus(finalLocation); 
		}
		
		double C = 3;
		System.out.println("Start SVM, wait...");
		SVM mySVM = new SVM(corpus, classNumber, featureSize, C);
		mySVM.crossValidation(10, corpus, classNumber);
		//ArrayList<_Doc> docs = corpus.getCollection();
		//What is the l in problem???
		
		//svm.svm_cross_validation(problem, parameter, int arg2, double[] arg3);
		//Shall we translate the arg parameters to the libsvm or construct the new data structure?
		System.out.println("Start training, wait...");
		
		System.out.println("Training finished!");
	}
}
