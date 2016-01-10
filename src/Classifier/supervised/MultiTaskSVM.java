package Classifier.supervised;

import java.util.ArrayList;
import java.util.Arrays;

import structures._PerformanceStat;
import structures._Review;
import structures._SparseFeature;
import structures._User;
import Classifier.supervised.liblinear.Feature;
import Classifier.supervised.liblinear.FeatureNode;
import Classifier.supervised.liblinear.Linear;
import Classifier.supervised.liblinear.Model;
import Classifier.supervised.liblinear.Parameter;
import Classifier.supervised.liblinear.Problem;
import Classifier.supervised.liblinear.SolverType;

public class MultiTaskSVM {
	double m_u; // Trade-off parameters between global model and individual model.
	double m_learningRate; // How many will be used as the training samples.
	
	int m_noTasks; // Total number of tasks(users).
	int m_featureNo; // Total number of features.
	int m_noXs;
	
	ArrayList<_User> m_users;
	Feature[][] m_Fvs; // Features of all tasks.
	double[] m_Ys; // Label of all samples.
	Model m_libModel; // Libmodel trained by liblinear.
	double m_bias; // Bias term for liblinear.
	
	_PerformanceStat[] m_perfStats; // Used to calculate the performance statictics.
	double[][] m_avgPRF; 
	
	public MultiTaskSVM(ArrayList<_User> users, int fn){
		m_u = 0.1;
		m_learningRate = 0.5;
		m_users = users;
		m_featureNo = fn + 1; //1 is for bias term.
		m_bias = 1;
		m_perfStats = new _PerformanceStat[m_users.size()];
		m_avgPRF = new double[2][3];
	}
	
	public void setTradeOffParam(double u){
		m_u = u;
	}
	
	public void init(){
		m_noTasks = m_users.size();
		for(int i=0; i<m_users.size(); i++)
			m_noXs += m_users.get(i).getReviews().size() * m_learningRate;// Get the total number of reviews.
	}
	
	public void train(){
		//Transfer all user reviews to instances recognized by SVM, indexed by users.
		int rid = 0, trainNo = 0;
		Feature [][] fvs = new Feature[m_noXs][];
		double[] ys = new double [m_noXs];
		ArrayList<_Review> reviews;
		//Two for loop to access the reviews, indexed by users.
		for(int i=0; i<m_users.size(); i++){
			reviews = m_users.get(i).getReviews();
			trainNo = (int) (reviews.size() * m_learningRate);
			for(int j=0; j < trainNo; j++){
				fvs[rid] = createLibLinearFV(reviews.get(j), i);
				ys[rid] = reviews.get(j).getYLabel();
				rid++;
			}
		}
		
		// Train a liblinear model based on all reviews.
		Problem libProblem = new Problem();
		libProblem.l = rid;
		libProblem.n = m_featureNo * (m_noTasks + 1);
		libProblem.x = fvs;
		libProblem.y = ys;
		libProblem.bias = m_bias;//bias term in liblinear.
		
		SolverType type = SolverType.L2R_L1LOSS_SVC_DUAL;//solver type: SVM
		double C = 1, EPS = 0.001;
		m_libModel = Linear.train(libProblem, new Parameter(type, C, EPS));
	}
	
	//create an instance of svm.
	public Feature[] createLibLinearFV(_Review r, int userIndex){
		
		_SparseFeature fv;
		_SparseFeature[] fvs = r.getSparse();
		Feature[] node = new Feature[fvs.length * 2];//0-th: x//sqrt(u); t-th: x.
		for(int i = 0; i < fvs.length; i++){
			fv = fvs[i];
			//Construct the global part of the training instance.
			node[i] = new FeatureNode(fv.getIndex() + 1, fv.getValue() / m_u);
			//Construct the t-th part of the training instance.
			node[i + fvs.length] = new FeatureNode(fv.getIndex() + 1 + m_featureNo * (userIndex + 1), fv.getValue());
		}
		return node;
	}
	
	//Use the each user's remaining reviews for testing.
	public void predict(){
		int trueL = 0, predL = 0, start = 0;
		int[][] TPTable = new int[2][2];
		ArrayList<_Review> reviews;
		_Review review;
		for(int i = 0; i < m_users.size(); i++) {
			reviews = m_users.get(i).getReviews();
			start = (int) (reviews.size() * m_learningRate);
			for(int j = start; j < reviews.size(); j++){
				review = reviews.get(j);
				predL = (int) Linear.predict(m_libModel, createLibLinearFV(review, i));
				trueL = review.getYLabel();
				TPTable[predL][trueL]++;
			}
			m_perfStats[i] = new _PerformanceStat(TPTable);
			clearTPTable(TPTable);
		}
	}
	
	public void clearTPTable(int[][] TPTable){
		for(int i=0; i<TPTable.length; i++)
			Arrays.fill(TPTable[i], 0);
	}
	
	//Accumulate the performance, accumulate all users.
	public void calcPerformance(){
		_PerformanceStat stat;
		for(int i=0; i<m_perfStats.length; i++){
			stat = m_perfStats[i];
			stat.calculatePRF();
			addOneUserPRF(stat.getPerformanceTable());
		}
		
		//Print out the precision/recall/F1.
		for(int i=0; i<m_avgPRF.length; i++){
			for(int j=0; j<m_avgPRF[0].length; j++)
				m_avgPRF[i][j] /= m_users.size();
		}
	}
	
	//Add one user's prf to the global prf.
	public void addOneUserPRF(double[][] prf) {
		if (prf.length == 0 || prf == null)
			return;
		if (prf.length != m_avgPRF.length)
			return;
		if (prf[0].length != m_avgPRF[0].length)
			return;

		for (int i = 0; i < prf.length; i++) {
			for (int j = 0; j < prf[i].length; j++)
				m_avgPRF[i][j] += prf[i][j];
		}
	}
	
	//Print out performance information.
	public void printPerformance() {
		System.out.format("\tprec\trecall\tF1\n");
		for (int i = 0; i < m_avgPRF.length; i++) {
			System.out.format("class %d\t", i);
			for (int j = 0; j < m_avgPRF[0].length; j++)
				System.out.format("%.4f\t", m_avgPRF[i][j]);
			System.out.println();
		}
	}
	
	public void batchTrainTest(){
		init();
		train();
		predict();
	}
}
