package Classifier.supervised;

import java.util.ArrayList;
import structures._Review;
import structures._SparseFeature;
import structures._User;
import structures._PerformanceStat.TestMode;
import structures._Review.rType;
import Classifier.supervised.liblinear.Feature;
import Classifier.supervised.liblinear.FeatureNode;
import Classifier.supervised.liblinear.Linear;
import Classifier.supervised.liblinear.Model;
import Classifier.supervised.liblinear.Parameter;
import Classifier.supervised.liblinear.Problem;
import Classifier.supervised.liblinear.SolverType;
import Classifier.supervised.modelAdaptation.ModelAdaptation;
import Classifier.supervised.modelAdaptation._AdaptStruct;

public class IndividualSVM extends ModelAdaptation {
	double m_C = 1.0; 
	boolean m_bias = true;
	Model m_libModel; // Libmodel trained by liblinear.
	
	SolverType m_solverType = SolverType.L2R_L1LOSS_SVC_DUAL;
	
	public IndividualSVM(int classNo, int featureSize){
		super(classNo, featureSize);
		m_testmode = TestMode.TM_batch;
	}
	
	@Override
	public String toString() {
		return String.format("Individual-SVM[C:%.3f,bias:%b]", m_C, m_bias);
	}

	@Override
	public void loadUsers(ArrayList<_User> userList) {
		m_userList = new ArrayList<_AdaptStruct>();
		for(_User user:userList) 
			m_userList.add(new _AdaptStruct(user));
		m_pWeights = new double[m_featureSize+1];		
	}
	
	public void setBias(boolean b){
		m_bias = b;
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
			trainSize = 0;
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
			
			// Train individual model for each user.
			Problem libProblem = new Problem();
			libProblem.l = trainSize;		
			libProblem.x = new Feature[trainSize][];
			libProblem.y = new double[trainSize];
			for(int i=0; i<trainSize; i++) {
				libProblem.x[i] = fvs.get(i);
				libProblem.y[i] = ys.get(i);
			}
			if (m_bias) {
				libProblem.n = m_featureSize + 1; // including bias term; global model + user models
				libProblem.bias = 1;// bias term in liblinear.
			} else {
				libProblem.n = m_featureSize;
				libProblem.bias = -1;// no bias term in liblinear.
			}
	
			m_libModel = Linear.train(libProblem, new Parameter(m_solverType, m_C, SVM.EPS));
			setPersonalizedModel(user);
		}
		return 0;
	}
	
	protected void setPersonalizedModel(_AdaptStruct u){
		u.setPersonalizedModel(m_libModel.getWeights());
	}	
	
	public Feature[] createLibLinearFV(_Review r, int userIndex){
		int fIndex; double fValue;
		_SparseFeature fv;
		_SparseFeature[] fvs = r.getSparse();
	
		Feature[] node;
		if(m_bias)
			node = new Feature[fvs.length + 1];//0-th: x//sqrt(u); t-th: x.
		else 
			node = new Feature[fvs.length];
		
		for(int i = 0; i < fvs.length; i++){
			fv = fvs[i];
			fIndex = fv.getIndex() + 1;//liblinear's feature index starts from one
			fValue = fv.getValue();
			
			//Construct the user part of the training instance.			
			node[i] = new FeatureNode(fIndex, fValue);
		}
		if (m_bias)//add the bias term		
			node[fvs.length] = new FeatureNode(m_featureSize + 1, 1.0);//user model's bias
		
		return node;
	}
	
	@Override
	protected void setPersonalizedModel() {
	}
}
