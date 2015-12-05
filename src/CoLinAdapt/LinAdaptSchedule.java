package CoLinAdapt;

import java.util.ArrayList;
import java.util.HashMap;
import structures.MyLinkedList;
import structures._Review;
import structures._User;

/****
 * In this class, we collect all adaptation users and try to schedule reviews and adaptation process.
 * @author lin
 */
public class LinAdaptSchedule {
	
	protected ArrayList<_User> m_users; //Mapping from users to models.
	protected HashMap<String, Integer> m_userIDIndexMap; // key: userID, value: index.
	protected String[] m_userIDs; // If we know index, we can get the userID.
	protected MyLinkedList<_Review> m_trainQueue;
	protected int m_featureNo, m_featureGroupNo;
	protected double[] m_globalWeights;
	protected int[] m_featureGroupIndexes;
	protected double[][] m_avgPRF;
	
	public LinAdaptSchedule(ArrayList<_User> users, int featureNo, int featureGroupNo, int[] featureGroupIndexes){
		m_users = users;
		m_userIDs = new String[users.size()];
		m_userIDIndexMap = new HashMap<String, Integer>();
		m_featureNo = featureNo;
		m_featureGroupNo = featureGroupNo;
		m_featureGroupIndexes = featureGroupIndexes;
		m_avgPRF = new double[2][3];
	}
	
	//Set the global weights.
	public void setGlobalWeights(double[] ws){
		m_globalWeights = ws;
	}

	//Fill in the user related information map and array.
	public void initSchedule(){
		_User user;
		for(int i=0; i<m_users.size(); i++){
			user = m_users.get(i);
			m_userIDs[i] = user.getUserID();
			m_userIDIndexMap.put(user.getUserID(), i);
			//Init each user's linAdapt model.
			user.initLinAdapt(m_featureGroupNo, m_featureNo, m_globalWeights, m_featureGroupIndexes); 
		}
	}
	
	//In this process, we collect one review from each user and train online.
	public void onlineTrain(){
		int predL, trueL;
		int[][] TPTable = new int[2][2];
//		int[] predLabels;
//		int[] trueLabels;
		LinAdapt model;
		ArrayList<_Review> reviews;
		
		//The review order doesn't matter, so we can adapt one user by one user.
		for(int i=0; i<m_users.size(); i++){
			model = m_users.get(i).getLinAdapt();
			reviews = m_users.get(i).getReviews();
//			predLabels = new int[reviews.size()];
//			trueLabels = new int[reviews.size()];
			for(int j=0; j<reviews.size(); j++){
				//Predict first.
				predL = model.predict(reviews.get(j));
				trueL = reviews.get(j).getYLabel();
				TPTable[predL][trueL]++;
//				predLabels[j] = predL;
//				trueLabels[j] = reviews.get(j).getYLabel();
				
				//Adapt based on the new review.
				ArrayList<_Review> trainSet = new ArrayList<_Review>();
				trainSet.add(reviews.get(j));
				model.train(trainSet);
			}
			model.setPerformanceStat(TPTable);
		}
	}
	
	//In batch mode, we use half of one user's reviews as training set.
	public void batchTrainTest(){
		LinAdapt model;
		ArrayList<_Review> reviews;
		int pivot = 0;
		
		//Traverse all users and train their models based on the half of their reviews.
		for(int i=0; i<m_users.size(); i++){
			model = m_users.get(i).getLinAdapt();
			reviews = m_users.get(i).getReviews();
			pivot = reviews.size()/2;
			//Split the reviews into two parts, one for training and another for testing.
			ArrayList<_Review> trainSet = new ArrayList<_Review>();
			ArrayList<_Review> testSet = new ArrayList<_Review>();
			for(int j=0; j<reviews.size(); j++){
				if(j < pivot)
					trainSet.add(reviews.get(j));
				else 
					testSet.add(reviews.get(j));
			}
			model.train(trainSet);//Train the model.
			model.test(testSet);
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
	
	//Accumulate the performance, accumulate all users.
	public void calcPerformance(){
		LinAdapt model;
		
		for(int i=0; i<m_users.size(); i++){
			model = m_users.get(i).getLinAdapt();
			model.m_perfStat.calculatePRF();
			addOneUserPRF(model.m_perfStat.getOneUserPRF());
		}
		
		for(int i=0; i<m_avgPRF.length; i++){
			for(int j=0; j<m_avgPRF[0].length; j++)
				m_avgPRF[i][j] /= m_users.size();
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
}
