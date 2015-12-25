package CoLinAdapt;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;

import structures.MyLinkedList;
import structures.MyPriorityQueue;
import structures._RankItem;
import structures._Review;
import structures._User;
import utils.Utils;

public class CoLinAdaptSchedule extends LinAdaptSchedule {
	double[] m_similarity;//It contains all user pair's similarity.
	
	public CoLinAdaptSchedule(ArrayList<_User> users, int featureNo, int featureGroupNo, int[] featureGroupIndexes){
		super(users, featureNo, featureGroupNo, featureGroupIndexes);
	}
	
	//Fill in the user related information map and array.
	public void initSchedule() {
		_User user;
		for (int i = 0; i < m_users.size(); i++) {
			user = m_users.get(i);
			m_userIDs[i] = user.getUserID();
			m_userIDIndexMap.put(user.getUserID(), i);
			user.initCoLinAdapt(m_featureGroupNo, m_featureNo, m_globalWeights, m_featureGroupIndexes); // Init each user's CoLinAdapt model.
		}
	}
	//Specify the neighbors of the current user.
	public void constructNeighborhood(){
		ArrayList<_User> neighbors;
		_User user;
		for(int i=0; i<m_users.size(); i++){
			user = m_users.get(i);
			neighbors = new ArrayList<_User>();
			for(int j=0; j<m_users.size(); j++){
				if(j != i)
					neighbors.add(m_users.get(j));
			}
			//For testing purpose, we use all others as neighbors.
			user.setNeighbors(neighbors); // Pass the references to the user as neighbors.
			user.setCoLinAdaptNeighbors(); //Pass neighbors to CoLinAdapt model.
		}
	}
	
	public void constructNeighborhood(int topK){
		_User user;
		ArrayList<_User> neighbors = new ArrayList<_User>();
		ArrayList<Integer> neighborIndexes = new ArrayList<Integer>();
		ArrayList<Double> neighborSims = new ArrayList<Double>();
		MyPriorityQueue<_RankItem> queue = new MyPriorityQueue<_RankItem>(topK);
		
		for(int i=0; i<m_users.size(); i++){
			user = m_users.get(i);
			for(int j=0; j<m_users.size(); j++){
				if(i != j)
					queue.add(new _RankItem(j, m_similarity[getIndex(i, j)]));//Sort all the neighbors based on similarity.
			}
			// Add the neighbors.
			for(_RankItem item: queue){
				neighbors.add(m_users.get(item.m_index));
				neighborIndexes.add(item.m_index);
				neighborSims.add(item.m_value);
			}
			user.setNeighbors(new ArrayList<_User>(neighbors));//Set the neighbors for the user.
			user.setCoLinAdaptNeighbors(); //Pass neighbors to the coLinAdapt model.
			user.setNeighborIndexes(new ArrayList<Integer>(neighborIndexes));
			user.setNeighborSims(new ArrayList<Double>(neighborSims));
			user.setCoLinAdpatNeighborSims();

			queue.clear();
			neighbors.clear();
			neighborSims.clear();
			neighborIndexes.clear();
		}
	}
	
	//Load the neighbors for each user from neighbor file.
	public void loadUserNeighbors(String filename, int topK){
		try{
			int userIndex = 0, neighborIndex = 0;;
			_User user;
			String neighborFile = String.format("%s_%d.txt", filename, topK);
			String line;
			String[] strs;
			ArrayList<_User> neighbors = new ArrayList<_User>();
			ArrayList<Double> neighborSims = new ArrayList<Double>();
			ArrayList<Integer> neighborIndexes = new ArrayList<Integer>();
			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(neighborFile), "UTF-8"));
			while((line = reader.readLine()) != null){
				strs = line.split(",");
				if(strs.length == 2*topK){
					for(int i=0; i<strs.length-1; i+=2){
						neighborIndex = Integer.valueOf(strs[i]);
						neighborIndexes.add(neighborIndex);
						neighbors.add(m_users.get(neighborIndex));
						neighborSims.add(Double.valueOf(strs[i+1]));
					}
				} else
					System.err.println("Wrong number of neighbors.");
				
				user = m_users.get(userIndex);
				
				user.setNeighborIndexes(new ArrayList<Integer>(neighborIndexes));
				user.setCoLinAdaptNeighborIndexes();
				
				user.setNeighbors(new ArrayList<_User>(neighbors));
				user.setCoLinAdaptNeighbors();
				
				user.setNeighborSims(new ArrayList<Double>(neighborSims));
				user.setCoLinAdpatNeighborSims();
				
				neighborIndexes.clear();
				neighbors.clear();
				neighborSims.clear();
				userIndex++;
			}
			reader.close();
			System.out.format("Neighbors for %d users are loaded.\n", m_users.size());
		} catch(IOException e){
			e.printStackTrace();
		}
	}
	//Write neighbors and sims for each user.
	public void writeUserNeighbors(String filename, int topK){
		try{
			_User user;
			int[] neighborIndexes;
			ArrayList<Double> neighborSims;
			String neighborFile = String.format("%s_%d.txt", filename, topK);
			PrintWriter writer = new PrintWriter(new File(neighborFile));
			//Write neighbor ID and sim into file.
			for(int i=0; i<m_users.size(); i++){
				user = m_users.get(i);
				neighborIndexes = user.getNeighborIndexes();
				neighborSims = user.getNeighborSims();
				if(neighborIndexes.length == neighborSims.size()){
					for(int j=0; j < neighborIndexes.length; j++){
						writer.format("%d,%.4f,", neighborIndexes[j], neighborSims.get(j));
					}
				}
				writer.write("\n");
			}
			writer.close();
			System.out.format("Finish writing neighbors for %d users.", m_users.size());
		} catch(IOException e){
			e.printStackTrace();
		}
	}
	
	//In the online mode, train them one by one and consider the order of the reviews.
	public void onlineTrain(){
		_Review tmp, next;
		int userIndex, predL;
		CoLinAdapt model;
		int count = 0;
		m_trainQueue = new MyLinkedList<_Review>();//We use this to maintain the review pool.
		//Construct the initial pool.
		for(_User u: m_users)
			m_trainQueue.add(u.getOneReview()); //Collect one review from each user.
		
		while(!m_trainQueue.isEmpty()){
			//Get the head review: the most recent one.
			tmp = m_trainQueue.poll(); 
			userIndex = m_userIDIndexMap.get(tmp.getUserID());
			next = m_users.get(userIndex).getOneReview();
			if(next != null)
				m_trainQueue.add(next);
			
			// Predict first.
			model = m_users.get(userIndex).getCoLinAdapt();
			model.setAs();
			predL = model.predict(tmp);
			model.addOnePredResult(predL, tmp.getYLabel());
			if(!model.train(tmp))
				m_failCount++;
			count++;
			if(count % 1000 == 0)
				System.out.print(".");
		}
		System.out.format("\n%d fails in online optimization.\n", m_failCount);
	}

	//In batch mode, we use half of one user's reviews as training set and we concatenate all users' reviews.
	public void batchTrainTest() {
		m_failCount = 0;
		SyncCoLinAdapt sync = new SyncCoLinAdapt(m_featureGroupNo, m_featureNo, m_globalWeights, m_featureGroupIndexes, m_users);
//		CoLinAdapt model;
		ArrayList<_Review> reviews;
		int pivot = 0;

		ArrayList<_Review> trainSet = new ArrayList<_Review>();
		ArrayList<_Review> testSet = new ArrayList<_Review>();
		
		// Traverse all users and train their models based on the half of their reviews.
		for (int i = 0; i < m_users.size(); i++) {
//			model = m_users.get(i).getCoLinAdapt();
			reviews = m_users.get(i).getReviews();
			pivot = reviews.size() / 2;
			// Split the reviews into two parts, one for training and another for testing.
			for (int j = 0; j < reviews.size(); j++) {
				if (j < pivot)
					trainSet.add(reviews.get(j));
				else
					testSet.add(reviews.get(j));
			}
		}
		sync.init();
		System.out.println("Start batch training....");
//		sync.setSimilarities(m_similarity);
		if(!sync.train(trainSet))
			m_failCount++;// Train the model.
		sync.test(testSet);
	}
	
	//Calculate each user's performance.
	public void calcPerformance(){
		for(int i=0; i<m_avgPRF.length; i++)
			Arrays.fill(m_avgPRF[i], 0);

		CoLinAdapt model;
		for(int i=0; i<m_users.size(); i++){
			model = m_users.get(i).getCoLinAdapt();
			model.m_perfStat.calculatePRF();
			addOneUserPRF(model.m_perfStat.getOneUserPRF());
		}
		for(int i=0; i<m_avgPRF.length; i++){
			for(int j=0; j<m_avgPRF[0].length; j++){
				m_avgPRF[i][j] /= m_users.size();
			}
		}
	}
	
	public double calcSim4TwoUsers(_User ui, _User uj){
		return 	Utils.cosine(ui.getSparse(), uj.getSparse());
	}
	
	public int getIndex(int i, int j){
		//Swap i and j.
		if(i < j){
			int t = j;
			j = i;
			i = t;
		}
		return i*(i-1)/2+j;
	}
	
	public void calcluateSimilarities(){
		m_similarity = new double[m_users.size()*(m_users.size()-1)/2];
		//Pre-compute the similarities among all users.
		for(int i=1; i<m_users.size(); i++){
			for(int j=0; j<i; j++)
				m_similarity[getIndex(i, j)] = calcSim4TwoUsers(m_users.get(i), m_users.get(j));
		}		
	}
}
