package Classifier;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;

import libsvm.svm;
import libsvm.svm_model;
import libsvm.svm_node;
import libsvm.svm_parameter;
import libsvm.svm_problem;
import structures._Corpus;
import structures._Doc;
import structures._SparseFeature;
import Analyzer.DocAnalyzer;

public class SVM extends BaseClassifier{
	svm_parameter m_param; //Define it to be global variable.
	//Constructor without give C.
	public SVM(_Corpus c, int classNumber, int featureSize){
		super(c, classNumber, featureSize);
		//Set default value of the param.
		this.m_param = new svm_parameter();
		// default values
		this.m_param.svm_type = svm_parameter.C_SVC;
		this.m_param.kernel_type = svm_parameter.LINEAR; // Hongning: linear kernel is the general choice for text classification
		this.m_param.degree = 1;
		this.m_param.gamma = 0; // 1/num_features
		this.m_param.coef0 = 0.2;
		this.m_param.nu = 0.5;
		this.m_param.cache_size = 100;
		this.m_param.C = 1;
		this.m_param.eps = 1e-6;
		this.m_param.p = 0.1;
		this.m_param.shrinking = 1;
		this.m_param.probability = 0;
		this.m_param.nr_weight = 0;
		this.m_param.weight_label = new int[0];// Hongning: why set it to 0-length array??
		this.m_param.weight = new double[0];
	}

	//Constructor with a given C.
	public SVM(_Corpus c, int classNumber, int featureSize, double C){
		super(c, classNumber, featureSize);
		// Set default value of the param.
		this.m_param = new svm_parameter();
		// default values
		this.m_param.svm_type = svm_parameter.C_SVC;
		this.m_param.kernel_type = svm_parameter.LINEAR; // Hongning: linear kernel is thegeneral choice for text classification
		this.m_param.degree = 1;
		this.m_param.gamma = 0; // 1/num_features
		this.m_param.coef0 = 0.2;
		this.m_param.nu = 0.5;
		this.m_param.cache_size = 100;
		this.m_param.C = C;
		this.m_param.eps = 1e-6;
		this.m_param.p = 0.1;
		this.m_param.shrinking = 1;
		this.m_param.probability = 0;
		this.m_param.nr_weight = 0;
		this.m_param.weight_label = new int[0];// Hongning: why set it to 0-length array??
		this.m_param.weight = new double[0];
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
			/* Train the data set to get the parameter.
			 * Jave do not accept same function name with same parameters, different return types.
			 * I have to add a new variable in the train.*/
			svm_model model = train(this.m_trainSet, true);
			test(this.m_testSet, model);
			this.m_trainSet.clear();
			//this.m_testSet.clear(); // why did you design it in this way?
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
	public svm_model train(ArrayList<_Doc> trainSet, boolean flag) {
		svm_model model = new svm_model();
		svm_problem problem = new svm_problem();
		problem.x = new svm_node[trainSet.size()][];
		problem.y = new double [trainSet.size()];		
		
		//Construct the svm_problem by enumerating all docs.
		int docId = 0, fid, fvSize = 0;
		for(_Doc temp:trainSet){
			svm_node[] instance = new svm_node[temp.getDocLength()];
			fid = 0;
			for(_SparseFeature fv:temp.getSparse()){
				instance[fid] = new svm_node();
				instance[fid].index = 1+fv.getIndex();
				instance[fid].value = fv.getNormValue();
				
				if (fvSize<instance[fid].index)
					fvSize = instance[fid].index;
				fid ++;
			}
			problem.x[docId] = instance;
			problem.y[docId] = 2.0 * temp.getYLabel() - 1;
			docId ++;
		}	
		this.m_param.gamma = 1.0/fvSize;//Set the gamma of parameter.
		problem.l = docId;
		model = svm.svm_train(problem, this.m_param);
		return model;
	}

	public void test(ArrayList<_Doc> testSet, svm_model model){
		double[][] TPTable = new double [this.m_classNo][this.m_classNo];
		double[][] PreRecOfOneFold = new double[this.m_classNo][2];
		//Construct the svm_problem by enumerating all docs.
		for (_Doc temp:testSet) {
			svm_node[] nodes = new svm_node[temp.getDocLength()];
			int fid = 0;
			for (_SparseFeature fv:temp.getSparse()) {
				nodes[fid] = new svm_node();
				nodes[fid].index = 1 + fv.getIndex();
				nodes[fid].value = fv.getNormValue();	
				fid++;
			}
			double result = svm.svm_predict(model, nodes);
			if (result>0)
				TPTable[1][temp.getYLabel()] += 1;
			else
				TPTable[0][temp.getYLabel()] += 1;
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
		String tokenModel = "data/Model/en-token.bin"; //Token model.
		String finalLocation = "/Users/lingong/Documents/Lin'sWorkSpace/IR_Base/SVM/SVMFinal.txt"; //The destination of storing the final features with stats.
		String featureLocation = "/Users/lingong/Documents/Lin'sWorkSpace/IR_Base/SVM/SVMSelectedFeatures.txt";
		
		String providedCV = "";
		//String featureSelection = "";
		//String providedCV = "Features.txt"; //Provided CV.
		String featureSelection = "IG"; //Feature selection method.
		double startProb = 0.0; // Used in feature selection, the starting point of the features.
		double endProb = 0.4; // Used in feature selection, the ending point of the feature.
		
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
			DocAnalyzer analyzer = new DocAnalyzer(tokenModel, classNumber, null, featureSelection, startProb, endProb);
			analyzer.LoadDirectory(folder, suffix); //Load all the documents as the data set.
			analyzer.featureSelection(featureLocation); //Select the features.
			
			System.out.println("Start loading files, wait...");
			DocAnalyzer analyzer_2 = new DocAnalyzer(tokenModel, classNumber, featureLocation, null);
			analyzer_2.LoadDirectory(folder, suffix);
			featureSize = analyzer.getFeatureSize();
			corpus = analyzer_2.returnCorpus(finalLocation); 
			
		} else if(!providedCV.isEmpty() && !featureSelection.isEmpty()){
			//Case 4: provided CV, feature selection.
			DocAnalyzer analyzer = new DocAnalyzer(tokenModel, classNumber, null, featureSelection, startProb, endProb);
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
//		corpus.save2File("data/FVs/fvector.dat");
		double C = 3;// The default value is 1.
		System.out.println("Start SVM, wait...");
		SVM mySVM = new SVM(corpus, classNumber, featureSize, C);
		mySVM.crossValidation(5, corpus, classNumber);
	}
}
