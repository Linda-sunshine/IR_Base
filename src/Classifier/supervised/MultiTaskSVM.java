package Classifier.supervised;

import java.util.ArrayList;
import java.util.Collection;

import Classifier.BaseClassifier;
import Classifier.semisupervised.CoLinAdapt._LinAdaptStruct;
import Classifier.supervised.liblinear.Feature;
import Classifier.supervised.liblinear.FeatureNode;
import Classifier.supervised.liblinear.Linear;
import Classifier.supervised.liblinear.Model;
import Classifier.supervised.liblinear.Parameter;
import Classifier.supervised.liblinear.Problem;
import Classifier.supervised.liblinear.SolverType;
import structures._Doc;
import structures._PerformanceStat;
import structures._Review;
import structures._Review.rType;
import structures._SparseFeature;
import structures._User;
import utils.Utils;

public class MultiTaskSVM extends BaseClassifier {
	double m_u = 0.1; // trade-off parameter between global model and individual model.
	double m_C = 1.0; // trade-off parameter for SVM training 
	
	ArrayList<_User> m_userList;	
	int m_userSize; // valid user size
	Model m_libModel; // Libmodel trained by liblinear.
	boolean m_bias = true; // whether use bias term in SVM; by default, we will use it
	
	public MultiTaskSVM(int classNo, int featureSize, ArrayList<_User> users){
		super(classNo, featureSize);
		
		m_userList = users;
	}
	
	public void setTradeOffParam(double u, double C){
		m_u = Math.sqrt(u);
		m_C = C;
	}
	
	public void setBias(boolean bias) {
		m_bias = bias;
	}
	
	@Override
	public void init(){
		m_userSize = 0;//need to get the total number of valid users to construct feature vector for MT-SVM
		for(int i=0; i<m_userList.size(); i++){
			_User user = m_userList.get(i);
			ArrayList<_Review> reviews = user.getReviews();
			boolean validUser = false;
			for(_Review r:reviews) {
				if (r.getType() == rType.ADAPTATION) {
					if (validUser==false) {
						validUser = true;
						m_userSize ++;
					}
				}
			}
			user.getPerfStat().clear(); // clear accumulate performance statistics
		}
	}
	
	@Override
	public void train(){
		init();
		
		//Transfer all user reviews to instances recognized by SVM, indexed by users.
		int trainSize = 0;
		ArrayList<Feature []> fvs = new ArrayList<Feature []>();
		ArrayList<Double> ys = new ArrayList<Double>();		
		
		//Two for loop to access the reviews, indexed by users.
		ArrayList<_Review> reviews;
		for(int i=0; i<m_userList.size(); i++){
			reviews = m_userList.get(i).getReviews();			
			for(_Review r:reviews) {
				if (r.getType() == rType.ADAPTATION) {//we will only use the adaptation data for this purpose
					fvs.add(createLibLinearFV(r, i));
					ys.add(new Double(r.getYLabel()));
					trainSize ++;
				}
			}			
		}
		
		// Train a liblinear model based on all reviews.
		Problem libProblem = new Problem();
		libProblem.l = trainSize;		
		libProblem.x = new Feature[trainSize][];
		libProblem.y = new double[trainSize];
		for(int i=0; i<trainSize; i++) {
			libProblem.x[i] = fvs.get(i);
			libProblem.y[i] = ys.get(i);
		}
		
		if (m_bias) {
			libProblem.n = (m_featureSize + 1) * (m_userSize + 1); // including bias term; global model + user models
			libProblem.bias = 1;// bias term in liblinear.
		} else {
			libProblem.n = m_featureSize * (m_userSize + 1);
			libProblem.bias = -1;// no bias term in liblinear.
		}
		
		SolverType type = SolverType.L2R_L1LOSS_SVC_DUAL;//solver type: SVM
		m_libModel = Linear.train(libProblem, new Parameter(type, m_C, SVM.EPS));
		
		setPersonalizedModel();
	}
	
	void setPersonalizedModel() {
		double[] weight = m_libModel.getWeights(), pWeight = new double[m_featureSize+1];//our model always assume the bias term
		int class0 = m_libModel.getLabels()[0];
		double sign = class0 > 0 ? 1 : -1;
		int userOffset = 0, globalOffset = m_bias?(m_featureSize+1)*m_userSize:m_featureSize*m_userSize;
		for(_User user:m_userList) {
			for(int i=0; i<m_featureSize; i++) {
				pWeight[i+1] = sign*(weight[globalOffset+i]/m_u + weight[userOffset+i]);
			}
			
			if (m_bias) {
				pWeight[0] = sign*(weight[globalOffset+m_featureSize]/m_u + weight[userOffset+m_featureSize]);
				userOffset += m_featureSize+1;
			} else
				userOffset += m_featureSize;
			
			user.setModel(pWeight, m_classNo, m_featureSize);//our model always assume the bias term
		}
	}
	
	//create a training instance of svm.
	//for MT-SVM feature vector construction: we put user models in front of global model
	public Feature[] createLibLinearFV(_Review r, int userIndex){
		int fIndex; double fValue;
		_SparseFeature fv;
		_SparseFeature[] fvs = r.getSparse();
		
		int userOffset, globalOffset;		
		Feature[] node;//0-th: x//sqrt(u); t-th: x.
		
		if (m_bias) {
			userOffset = (m_featureSize + 1) * userIndex;
			globalOffset = (m_featureSize + 1) * m_userSize;
			node = new Feature[(1+fvs.length) * 2];
		} else {
			userOffset = m_featureSize * userIndex;
			globalOffset = m_featureSize * m_userSize;
			node = new Feature[fvs.length * 2];
		}
		
		for(int i = 0; i < fvs.length; i++){
			fv = fvs[i];
			fIndex = fv.getIndex() + 1;//liblinear's feature index starts from one
			fValue = fv.getValue();
			
			//Construct the user part of the training instance.			
			node[i] = new FeatureNode(userOffset + fIndex, fValue);
			
			//Construct the global part of the training instance.
			if (m_bias)
				node[i + fvs.length + 1] = new FeatureNode(globalOffset + fIndex, fValue/m_u); // global model's bias term has to be moved to the last
			else
				node[i + fvs.length] = new FeatureNode(globalOffset + fIndex, fValue/m_u); // global model's bias term has to be moved to the last
		}
		
		if (m_bias) {//add the bias term		
			node[fvs.length] = new FeatureNode((m_featureSize + 1) * (userIndex + 1), 1.0);//user model's bias
			node[2*fvs.length+1] = new FeatureNode((m_featureSize + 1) * (m_userSize + 1), 1.0 / m_u);//global model's bias
		}
		return node;
	}
	
	//Use the each user's remaining reviews for testing.
	@Override
	public double test(){
		int trueL = 0, predL = 0, count = 0;
		_PerformanceStat userPerfStat;
		double[] macroF1 = new double[m_classNo];
		
		for(int i=0; i<m_userList.size(); i++) {
			_User user = m_userList.get(i);
			userPerfStat = user.getPerfStat();
			
			for(_Review r:user.getReviews()){
				if (r.getType() != rType.TEST)
					continue;
				predL = user.predict(r);
				trueL = r.getYLabel();
				userPerfStat.addOnePredResult(predL, trueL);
			}
			m_microStat.accumulateConfusionMat(userPerfStat);
			
			userPerfStat.calculatePRF();			
			for(int n=0; n<m_classNo; n++)
				macroF1[n] += userPerfStat.getF1(n);
			
			count ++;
		}
		calcMicroPerfStat();
		
		// macro average
		System.out.println("\nMacro F1:");
		for(int i=0; i<m_classNo; i++)
			System.out.format("Class %d: %.3f\t", i, macroF1[i]/count);
		System.out.println();
		return Utils.sumOfArray(macroF1);
	}

	@Override
	public void train(Collection<_Doc> trainSet) {
		System.err.println("[Error]train(Collection<_Doc> trainSet) is not implemented in MultiTaskSVM!");
		System.exit(-1);
	}

	@Override
	public int predict(_Doc doc) {//predict by global model		
		return -1;
	}

	@Override
	public double score(_Doc d, int label) {//prediction score by global model
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	protected void debug(_Doc d) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void saveModel(String modelLocation) {
		// TODO Auto-generated method stub
		
	}
}