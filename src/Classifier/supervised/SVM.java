package Classifier.supervised;
import java.io.File;
import java.io.IOException;
import java.util.Collection;

import structures._Corpus;
import structures._Doc;
import structures._SparseFeature;
import Classifier.BaseClassifier;
import Classifier.supervised.liblinear.Feature;
import Classifier.supervised.liblinear.FeatureNode;
import Classifier.supervised.liblinear.Linear;
import Classifier.supervised.liblinear.Model;
import Classifier.supervised.liblinear.Parameter;
import Classifier.supervised.liblinear.Problem;
import Classifier.supervised.liblinear.SolverType;


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
	Parameter m_libParameter;
	
	//Constructor without give C.
	public SVM(_Corpus c, int classNumber, int featureSize){
		super(c, classNumber, featureSize);
		//Set default value of the param.
		m_libParameter = new Parameter(SolverType.L2R_L2LOSS_SVC_DUAL, 1.0, 0.01);
	}

	//Constructor with a given C.
	public SVM(_Corpus c, int classNumber, int featureSize, double C, double eps){
		super(c, classNumber, featureSize);
		// Set default value of the param.
		m_libParameter = new Parameter(SolverType.L2R_L2LOSS_SVC_DUAL, C, eps);
	}
	
	@Override
	public String toString() {
		return String.format("SVM[C:%d, F:%d, T:%s]", m_classNo, m_featureSize, m_libParameter.getSolverType());
	}
	
	@Override
	protected void init() {
		//no need to initiate, libSVM will take care of it
	}
	
	protected Feature[] createFV(_Doc doc) {
		Feature[] node = new Feature[doc.getDocLength()]; 
		int fid = 0;
		for(_SparseFeature fv:doc.getSparse())
			node[fid++] = new FeatureNode(1 + fv.getIndex(), fv.getValue());//svm's feature index starts from 1
		return node;
	}
	
	@Override
	public void train(Collection<_Doc> trainSet) {
		Feature[][] fvs = new Feature[trainSet.size()][];
		double[] y = new double[trainSet.size()];
		
		int fid = 0;
		for(_Doc d:trainSet) {
			fvs[fid] = createFV(d);
			y[fid] = d.getYLabel();
			fid ++;
		}
		
		Problem libProblem = new Problem();
		libProblem.l = fid;
		libProblem.n = m_featureSize;
		libProblem.x = fvs;
		libProblem.y = y;
		m_libModel = Linear.train(libProblem, m_libParameter);
	}
	
	@Override
	public int predict(_Doc doc) {
		return (int)Linear.predict(m_libModel, createFV(doc));
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
