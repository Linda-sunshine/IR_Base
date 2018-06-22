package Classifier.supervised.modelAdaptation;

import java.util.ArrayList;

import Classifier.supervised.SVM;
import Classifier.supervised.liblinear.Feature;
import Classifier.supervised.liblinear.FeatureNode;
import Classifier.supervised.liblinear.Linear;
import Classifier.supervised.liblinear.Model;
import Classifier.supervised.liblinear.Parameter;
import Classifier.supervised.liblinear.Problem;
import Classifier.supervised.liblinear.SolverType;
import structures._PerformanceStat.TestMode;
import structures._Review;
import structures._Doc.rType;
import structures._SparseFeature;
import structures._User;

public class MultiTaskSVM extends ModelAdaptation {
	double m_u = 1.0; // trade-off parameter between global model and individual model.
	double m_C = 1.0; // trade-off parameter for SVM training 
	
	Model m_libModel; // Libmodel trained by liblinear.
	boolean m_bias = true; // whether use bias term in SVM; by default, we will use it
	
	public MultiTaskSVM(int classNo, int featureSize){
		super(classNo, featureSize, null, null);
		
		// the only test mode for MultiTaskSVM is batch
		m_testmode = TestMode.TM_batch;
	}
	
	@Override
	public String toString() {
		return String.format("MT-SVM[mu:%.3f,C:%.3f,bias:%b]", m_u, m_C, m_bias);
	}
	
	@Override
	public void loadUsers(ArrayList<_User> userList) {
		m_userList = new ArrayList<_AdaptStruct>();
		for(_User user:userList) 
			m_userList.add(new _AdaptStruct(user));
		m_pWeights = new double[m_featureSize+1];
	}
	
	public void setTradeOffParam(double u, double C){
		m_u = Math.sqrt(u);
		m_C = C;
	}
	
	public void setBias(boolean bias) {
		m_bias = bias;
	}
	
	@Override
	public double train() {
		init();
		
		//Transfer all user reviews to instances recognized by SVM, indexed by users.
		int trainSize = 0, validUserIndex = 0;
		ArrayList<Feature []> fvs = new ArrayList<Feature []>();
		ArrayList<Double> ys = new ArrayList<Double>();		
		
		//Two for loop to access the reviews, indexed by users.
		ArrayList<_Review> reviews;
		for(_AdaptStruct user:m_userList){
			reviews = user.getReviews();		
			boolean validUser = false;
			for(_Review r:reviews) {				
				if (r.getType() == rType.ADAPTATION) {//we will only use the adaptation data for this purpose
					fvs.add(createLibLinearFV(r, validUserIndex));
					ys.add(new Double(r.getYLabel()));
					trainSize ++;
					validUser = true;
				}
			}
			
			if (validUser)
				validUserIndex ++;
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
		
		return 0;
	}
	
	public void setLibProblemDimension(Problem libProblem){
		if (m_bias) {
			libProblem.n = (m_featureSize + 1) * (m_userSize + 1); // including bias term; global model + user models
			libProblem.bias = 1;// bias term in liblinear.
		} else {
			libProblem.n = m_featureSize * (m_userSize + 1);
			libProblem.bias = -1;// no bias term in liblinear.
		}
	}
	@Override
	protected void setPersonalizedModel() {
		double[] weight = m_libModel.getWeights();//our model always assume the bias term
		int class0 = m_libModel.getLabels()[0];
		double sign = class0 > 0 ? 1 : -1, block=m_personalized?1:0;//disable personalized model when required
		int userOffset = 0, globalOffset = m_bias?(m_featureSize+1)*m_userSize:m_featureSize*m_userSize;
		for(_AdaptStruct user:m_userList) {
			if (user.getAdaptationSize()>0) {
				for(int i=0; i<m_featureSize; i++) 
					m_pWeights[i+1] = sign*(weight[globalOffset+i]/m_u + block*weight[userOffset+i]);
				
				if (m_bias) {
					m_pWeights[0] = sign*(weight[globalOffset+m_featureSize]/m_u + block*weight[userOffset+m_featureSize]);
					userOffset += m_featureSize+1;
				} else
					userOffset += m_featureSize;
			} else {
				for(int i=0; i<m_featureSize; i++) // no personal model since no adaptation data
					m_pWeights[i+1] = sign*weight[globalOffset+i]/m_u;
				
				if (m_bias)
					m_pWeights[0] = sign*weight[globalOffset+m_featureSize]/m_u;
			}
			
			user.setPersonalizedModel(m_pWeights);//our model always assume the bias term
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
}
