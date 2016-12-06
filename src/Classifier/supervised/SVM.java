package Classifier.supervised;
import java.awt.List;
import java.io.*;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;

import Classifier.BaseClassifier;
import Classifier.supervised.liblinear.Feature;
import Classifier.supervised.liblinear.Linear;
import Classifier.supervised.liblinear.Model;
import Classifier.supervised.liblinear.Parameter;
import Classifier.supervised.liblinear.Problem;
import Classifier.supervised.liblinear.SolverType;
import structures._Corpus;
import structures._Doc;
import utils.Utils;


/**
 * old implementation is based on libsvm, which is very slow and inefficient
 * @author hongning
 *
 */
//public class SVM extends BaseClassifier {
//	svm_parameter m_param; //Define it to be global variable.
//	svm_model m_model;
//	
//	//Constructor without give C.
//	public SVM(_Corpus c, int classNumber, int featureSize){
//		super(c, classNumber, featureSize);
//		//Set default value of the param.
//		m_param = new svm_parameter();
//		m_param.svm_type = svm_parameter.C_SVC;
//		m_param.kernel_type = svm_parameter.LINEAR; // Hongning: linear kernel is the general choice for text classification
//		m_param.degree = 1;
//		m_param.gamma = 0; // 1/num_features
//		m_param.coef0 = 0.2;
//		m_param.nu = 0.5;
//		m_param.cache_size = 100;
//		m_param.C = 1;
//		m_param.eps = 1e-6;
//		m_param.p = 0.1;
//		m_param.shrinking = 1;
//		m_param.probability = 0;
//		m_param.nr_weight = 0;
//		m_param.weight_label = new int[0];// Hongning: why set it to 0-length array??
//		m_param.weight = new double[0];
//	}
//
//	//Constructor with a given C.
//	public SVM(_Corpus c, int classNumber, int featureSize, double C){
//		super(c, classNumber, featureSize);
//		// Set default value of the param.
//		m_param = new svm_parameter();
//		m_param.svm_type = svm_parameter.C_SVC;
//		m_param.kernel_type = svm_parameter.LINEAR; // Hongning: linear kernel is thegeneral choice for text classification
//		m_param.degree = 1;
//		m_param.gamma = 0; // 1/num_features
//		m_param.coef0 = 0.2;
//		m_param.nu = 0.5;
//		m_param.cache_size = 100;
//		m_param.C = C;
//		m_param.eps = 1e-6;
//		m_param.p = 0.1;
//		m_param.shrinking = 1;
//		m_param.probability = 0;
//		m_param.nr_weight = 0;
//		m_param.weight_label = new int[0];// Hongning: why set it to 0-length array??
//		m_param.weight = new double[0];
//	}
//	
//	@Override
//	public String toString() {
//		return String.format("SVM[C:%d, F:%d, T:%d]", m_classNo, m_featureSize, m_param.svm_type);
//	}
//	
//	@Override
//	protected void init() {
//		//no need to initiate, libSVM will take care of it
//	}
//	
//	protected svm_node[] createSample(_Doc doc) {
//		svm_node[] node = new svm_node[doc.getDocLength()]; 
//		int fid = 0;
//		for(_SparseFeature fv:doc.getSparse()){
//			node[fid] = new svm_node();
//			node[fid].index = 1 + fv.getIndex();//svm's feature index starts from 1
//			node[fid].value = fv.getValue();
//			fid ++;
//		}
//		return node;
//	}
//	
//	@Override
//	public void train(Collection<_Doc> trainSet) {
//		svm_problem problem = new svm_problem();
//		problem.x = new svm_node[trainSet.size()][];
//		problem.y = new double [trainSet.size()];		
//		
//		//Construct the svm_problem by enumerating all docs.
//		int docId = 0;
//		for(_Doc doc : trainSet){
//			problem.x[docId] = createSample(doc);
//			problem.y[docId] = doc.getYLabel();
//			docId ++;
//		}	
//		m_param.gamma = 1.0/m_featureSize;//Set the gamma of parameter.
//		problem.l = docId;
//		m_model = Classifier.supervised.libsvm.svm.svm_train(problem, m_param);
//	}
//	
//	@Override
//	public int predict(_Doc doc) {
//		return (1+(int)Classifier.supervised.libsvm.svm.svm_predict(m_model, createSample(doc)))/2;
//	}
//	
//	@Override
//	protected void debug(_Doc d) {} // need to digest into libsvm's implementation detail
//	
//	//Save the parameters for classification.
//	@Override
//	public void saveModel(String modelLocation){
//		try {
//			Classifier.supervised.libsvm.svm.svm_save_model(modelLocation, m_model);
//		} catch (IOException e) {
//			e.printStackTrace();
//		}
//	}
//}

/**
 * new implementation is based on liblinear, which is much more efficient
 * @author hongning
 *
 */
public class SVM extends BaseClassifier {
	Model m_libModel;
	SolverType m_type = SolverType.L2R_L1LOSS_SVC_DUAL;
	double m_C;
	final static public double EPS = 0.001;
	
	//Constructor with a given C.
	public SVM(_Corpus c, double C){
		super(c);
		// Set default value of the param.
		m_C = C;
	}

	//Constructor with a given C.
	public SVM(int classNo, int featureSize, double C){
		super(classNo, featureSize);
		// Set default value of the param.
		m_C = C; 
	}
		
	@Override
	public String toString() {
		return String.format("SVM[C:%d, F:%d, T:%s, c:%.3f]", m_classNo, m_featureSize, m_type, m_C);
	}
	
	@Override
	protected void init() {
		//no need to initiate, libSVM will take care of it
	}

	//k-fold Cross Validation.
	public void crossValidation(int k, _Corpus c){
		try {
			if (m_debugOutput!=null){
				m_debugWriter = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(m_debugOutput, false), "UTF-8"));
				m_debugWriter.write(this.toString() + "\n");
			}
//				c.shuffle(k);

//				int[] masks = c.getMasks();
			ArrayList<_Doc> docs = c.getCollection();
			double[] accuracyArray = new double[k];
			//Use this loop to iterate all the ten folders, set the train set and test set.
			for (int i = 0; i < k; i++) {
//					for (int j = 0; j < masks.length; j++) {


				//more for training
//						if(masks[j]==i)
//							m_testSet.add(docs.get(j));
//						else
//							m_trainSet.add(docs.get(j));
//						if( masks[j]==(i+1)%k || masks[j]==(i+2)%k ) // || masks[j]==(i+3)%k
//							m_trainSet.add(docs.get(j));
//						else
//							m_testSet.add(docs.get(j));

//						//more for testing
//						if(masks[j]==i)
//							m_trainSet.add(docs.get(j));
//						else
//							m_testSet.add(docs.get(j));
//					}
				double trainingProportion = 0.01;
				splitData(c, trainingProportion);
				System.out.println("train set\t"+m_trainSet.size()+"\t test set\t"+m_testSet.size());
				long start = System.currentTimeMillis();
				train();
				double accuracy = test();
				accuracyArray[i] = accuracy;
				System.out.format("%s Train/Test finished in %.2f seconds with accuracy %.4f and F1 (%s)...\n", this.toString(), (System.currentTimeMillis()-start)/1000.0, accuracy, getF1String());
				m_trainSet.clear();
				m_testSet.clear();
			}
			calculateMeanVariance(m_precisionsRecalls);

			double accuracyVar = Utils.getVariance(accuracyArray);
			double accuracyAvg = Utils.getMean(accuracyArray);
			System.out.println("accuracy avg:\t"+accuracyAvg+"\t var:\t"+accuracyVar);

			if (m_debugOutput!=null)
				m_debugWriter.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	protected void splitData(_Corpus c, double trainingProportion){
		HashMap<Integer, ArrayList<_Doc>> labelDocMap = new HashMap<Integer, ArrayList<_Doc>>();
		for(_Doc d:c.getCollection()){
			int yLabel = d.getYLabel();
			if(labelDocMap.containsKey(yLabel)){
				ArrayList<_Doc>labelDocList = labelDocMap.get(yLabel);
				labelDocList.add(d);
			}else{
				ArrayList<_Doc> labelDocList = new ArrayList<_Doc>();
				labelDocList.add(d);
				labelDocMap.put(yLabel, labelDocList);
			}
		}
		for(int yLabel:labelDocMap.keySet()){
			ArrayList<_Doc> labelDocList = labelDocMap.get(yLabel);
			Collections.shuffle(labelDocList);
		}


		for(int yLabel:labelDocMap.keySet()){
			ArrayList<_Doc>labelDocList = labelDocMap.get(yLabel);

			int trainingNum = (int)(labelDocList.size()*trainingProportion);

			for(int i=0; i<trainingNum; i++)
				m_trainSet.add(labelDocList.get(i));

			for(int i=trainingNum; i<labelDocList.size(); i++)
				m_testSet.add(labelDocList.get(i));
		}


	}

	@Override
	public double train(Collection<_Doc> trainSet) {
		m_libModel = libSVMTrain(trainSet, 1+m_featureSize, m_type, m_C, 1);
		return 0;
	}
	
	public static Model libSVMTrain(Collection<_Doc> trainSet, int fSize, SolverType type, double C, double bias) {
		Feature[][] fvs = new Feature[trainSet.size()][];
		double[] y = new double[trainSet.size()];
		
		int fid = 0; // file id
//		ArrayList<Integer> labelList = new ArrayList<Integer>();
		for(_Doc d:trainSet) {
			if (bias>0)
				fvs[fid] = Utils.createLibLinearFV(d, fSize);
			else
				fvs[fid] = Utils.createLibLinearFV(d, 0);
			
//			if (!labelList.contains(d.getYLabel()))
//				labelList.add(d.getYLabel());
			y[fid] = d.getYLabel();
			fid ++;
		}
//		System.out.println("total documents\t"+fid);
		
		Problem libProblem = new Problem();
		libProblem.l = fid;
		libProblem.n = bias>=0?1+fSize:fSize;
		libProblem.x = fvs;
		libProblem.y = y;
		libProblem.bias = bias;
		
		return Linear.train(libProblem, new Parameter(type, C, SVM.EPS));
	}
	
	public static Model libSVMTrain(ArrayList<Feature[]> featureArray, ArrayList<Integer> targetArray,
			int fSize, SolverType type, double C, double bias) {
		
		Feature[][] featureMatrix = new Feature[featureArray.size()][];
		double[] targetMatrix = new double[targetArray.size()];
		for(int i = 0; i < featureArray.size(); i++){
			featureMatrix[i] = featureArray.get(i);
			targetMatrix[i] = targetArray.get(i);
		}
		
		Problem libProblem = new Problem();
		libProblem.l = featureMatrix.length;
		libProblem.n = fSize;
		libProblem.x = featureMatrix;
		libProblem.y = targetMatrix;
		libProblem.bias = bias;
		
		return Linear.train(libProblem, new Parameter(type, C, SVM.EPS));
	}
	
	@Override
	public int predict(_Doc doc) {
		return (int)Linear.predict(m_libModel, Utils.createLibLinearFV(doc, m_featureSize));
	}
	
	@Override
	public double score(_Doc doc, int label) {
		return Linear.predictValue(m_libModel, Utils.createLibLinearFV(doc, m_featureSize), label);
	}
	
	@Override
	protected void debug(_Doc d) {} // need to digest into libsvm's implementation detail
	
	//Save the parameters for classification.
	@Override
	public void saveModel(String modelLocation){
		try {
			Linear.saveModel(new File(modelLocation), m_libModel);
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
}
