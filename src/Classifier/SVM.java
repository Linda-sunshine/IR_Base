package Classifier;
import java.util.ArrayList;
import libsvm.svm;
import libsvm.svm_model;
import libsvm.svm_node;
import libsvm.svm_parameter;
import libsvm.svm_problem;
import structures._Corpus;
import structures._Doc;
import structures._SparseFeature;

public class SVM extends BaseClassifier{
	svm_parameter m_param; //Define it to be global variable.
	//Constructor without give C.
	public SVM(_Corpus c, int classNumber, int featureSize){
		super(c, classNumber, featureSize);
		//Set default value of the param.
		m_param = new svm_parameter();
		m_param.svm_type = svm_parameter.C_SVC;
		m_param.kernel_type = svm_parameter.LINEAR; // Hongning: linear kernel is the general choice for text classification
		m_param.degree = 1;
		m_param.gamma = 0; // 1/num_features
		m_param.coef0 = 0.2;
		m_param.nu = 0.5;
		m_param.cache_size = 100;
		m_param.C = 1;
		m_param.eps = 1e-6;
		m_param.p = 0.1;
		m_param.shrinking = 1;
		m_param.probability = 0;
		m_param.nr_weight = 0;
		m_param.weight_label = new int[0];// Hongning: why set it to 0-length array??
		m_param.weight = new double[0];
	}

	//Constructor with a given C.
	public SVM(_Corpus c, int classNumber, int featureSize, double C){
		super(c, classNumber, featureSize);
		// Set default value of the param.
		m_param = new svm_parameter();
		m_param.svm_type = svm_parameter.C_SVC;
		m_param.kernel_type = svm_parameter.LINEAR; // Hongning: linear kernel is thegeneral choice for text classification
		m_param.degree = 1;
		m_param.gamma = 0; // 1/num_features
		m_param.coef0 = 0.2;
		m_param.nu = 0.5;
		m_param.cache_size = 100;
		m_param.C = C;
		m_param.eps = 1e-6;
		m_param.p = 0.1;
		m_param.shrinking = 1;
		m_param.probability = 0;
		m_param.nr_weight = 0;
		m_param.weight_label = new int[0];// Hongning: why set it to 0-length array??
		m_param.weight = new double[0];
	}
	
	//k-fold Cross Validation.
	public void crossValidation(int k, _Corpus c){
		c.shuffle(k);
		int[] masks = c.getMasks();
		ArrayList<_Doc> docs = c.getCollection();
		//Use this loop to iterate all the ten folders, set the train set and test set.
		for (int i = 0; i < k; i++) {
			for (int j = 0; j < masks.length; j++) {
				if( masks[j]==i ) m_testSet.add(docs.get(j));
				else m_trainSet.add(docs.get(j));
			}
			//Train the data set to get the parameter.
			svm_model model = trainSVM();
			testSVM(model);
			m_trainSet.clear();
			m_testSet.clear();
		}
		calculateMeanVariance(m_precisionsRecalls);	
	}
	// Train the data set.
	public svm_model trainSVM() {
		svm_model model = new svm_model();
		svm_problem problem = new svm_problem();
		problem.x = new svm_node[m_trainSet.size()][];
		problem.y = new double [m_trainSet.size()];		
		
		//Construct the svm_problem by enumerating all docs.
		int docId = 0, fid, fvSize = 0;
		for(_Doc temp : m_trainSet){
			svm_node[] instance = new svm_node[temp.getDocLength()]; //this doc length is the number of sparse vectors.
			fid = 0;
			for(_SparseFeature fv:temp.getSparse()){
				instance[fid] = new svm_node();
				instance[fid].index = 1+fv.getIndex();
				instance[fid].value = fv.getValue();
				
				if (fvSize<instance[fid].index)
					fvSize = instance[fid].index;
				fid ++;
			}
			problem.x[docId] = instance;
			problem.y[docId] = temp.getYLabel();
			docId ++;
		}	
		m_param.gamma = 1.0/fvSize;//Set the gamma of parameter.
		problem.l = docId;
		model = svm.svm_train(problem, m_param);
		return model;
	}

	public void testSVM(svm_model model){
		//Construct the svm_problem by enumerating all docs.
		for (_Doc temp: m_testSet) {
			svm_node[] nodes = new svm_node[temp.getDocLength()]; //this doc length is the number of sparse vectors.
			int fid = 0;
			for (_SparseFeature fv:temp.getSparse()) {
				nodes[fid] = new svm_node();
				nodes[fid].index = 1 + fv.getIndex();
				nodes[fid].value = fv.getValue();	
				fid++;
			}
			int result = (int)svm.svm_predict(model, nodes);
			m_TPTable[(result + 1)/2][temp.getYLabel()] += 1;
		}
		m_PreRecOfOneFold = calculatePreRec(m_TPTable);
		m_precisionsRecalls.add(m_PreRecOfOneFold);
	}
	
	@Override
	public void train(){}
	public void test(){}
}
