package CoLinAdapt;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;

import org.apache.commons.math.optimization.GoalType;
import org.apache.commons.math.optimization.OptimizationException;
import org.apache.commons.math.optimization.RealPointValuePair;
import org.apache.commons.math.optimization.linear.LinearConstraint;
import org.apache.commons.math.optimization.linear.LinearObjectiveFunction;
import org.apache.commons.math.optimization.linear.Relationship;
import org.apache.commons.math.optimization.linear.SimplexSolver;

import structures.MyLinkedList;
import structures.MyPriorityQueue;
import structures._RankItem;
import structures._Review;
import structures._User;
import utils.Utils;

public class CoLinAdaptSchedule extends LinAdaptSchedule {
	double[] m_similarity;//It contains all user pair's similarity.
	double m_eta3 = 0.5;// scale for R2.
	double m_eta4 = 0.5; // shift for R2.
	
	//Parameters for simplex optimization.
	LinearObjectiveFunction m_LPObj; // The linear objective function.
	Collection m_LPConstrains = new ArrayList();
	RealPointValuePair m_solution;
	
	public CoLinAdaptSchedule(ArrayList<_User> users, int featureNo, int featureGroupNo, int[] featureGroupIndexes){
		super(users, featureNo, featureGroupNo, featureGroupIndexes);
	}
	
	public void setCoefficients4R2(double a4r2, double b4r2){
		m_eta3 = a4r2;
		m_eta4 = b4r2;
	}
	//Fill in the user related information map and array.
	public void initSchedule() {
		_User user;
		for (int i = 0; i < m_users.size(); i++) {
			user = m_users.get(i);
			m_userIDs[i] = user.getUserID();
			m_userIDIndexMap.put(user.getUserID(), i);
			user.initCoLinAdapt(i, m_featureGroupNo, m_featureNo, m_globalWeights, m_featureGroupIndexes); // Init each user's CoLinAdapt model.
			user.setCoefficients(m_eta1, m_eta2, m_eta3, m_eta4); //Set the coefficients for the model.
			user.setCoLinAdaptSimilarity(m_similarity);
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
//			user.setNeighborSims(new ArrayList<Double>(neighborSims));
//			user.setCoLinAdpatNeighborSims();

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
				
//				user.setNeighborSims(new ArrayList<Double>(neighborSims));
//				user.setCoLinAdpatNeighborSims();
				
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
		boolean flag;
		
		int[] neighborIndexes;
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
			if(!(flag = model.train(tmp)))
				m_failCount++;
			count++;
			if(count % 1000 == 0)
				System.out.print(".");
//			if(flag){
//				//Train the similarity of the neighbors.
//				m_newSimilarity = trainSimilarity(model.getR2Vct4LP());
//				if(m_newSimilarity != null){
//					System.out.print("lp.");
//					//Update the neighbor similarity.
//					neighborIndexes = m_users.get(userIndex).getNeighborIndexes();
//					for(int i=0; i<neighborIndexes.length; i++)
//						m_similarity[getIndex(userIndex, neighborIndexes[i])] = m_newSimilarity[i];
//				}
//			}
		}
		System.out.format("%d fails in online optimization.\n", m_failCount);
	}
	
//	//Use linear programming to train the similairty among current user and neighbors.
//	public double[] trainSimilarity(double[] R2Vct) {
//		try{
//			double[] solution = new double[R2Vct.length];
//			m_LPObj = new LinearObjectiveFunction(R2Vct, 0);
//			//Constrains:
//			double[][] constrains = new double[R2Vct.length][R2Vct.length];
//			for(int i=0; i<R2Vct.length; i++){
//				constrains[i][i] = 1;
//				m_LPConstrains.add(new LinearConstraint(constrains[i], Relationship.GEQ, 0));
//			}
//			double[] constrain = new double[R2Vct.length];
//			Arrays.fill(constrain, 1);
//			m_LPConstrains.add(new LinearConstraint(constrain, Relationship.EQ, 1));
//		
//			m_solution = new SimplexSolver().optimize(m_LPObj, m_LPConstrains, GoalType.MINIMIZE, false);
//			solution = m_solution.getPoint();
//			return solution;
//		} catch(OptimizationException e){
////			e.printStackTrace();
//			return null;
//		}
//	}

//	public double[] trainSimilarity(){
//		try{
//			int topK = m_users.get(0).getNeighbors().size();
//			double[] solution = new double[m_users.size() * topK];
//			m_LPObj = new LinearObjectiveFunciton();
//		} catch(OptimizationException e){
//			return null;
//		}
//	}
	//In batch mode, we use half of one user's reviews as training set and we concatenate all users' reviews.
	public void batchTrainTest() {
		m_failCount = 0;
		SyncCoLinAdapt sync = new SyncCoLinAdapt(m_featureGroupNo, m_featureNo, m_globalWeights, m_featureGroupIndexes, m_users, m_similarity);
//		CoLinAdapt model;
		sync.setCoefficients(m_eta1, m_eta2, m_eta3, m_eta4);
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
		if(!sync.train(trainSet))
			m_failCount++;// Train the model.
		
		int iterCount = 0;
		double simDiff = 100;
		double[] preSimilarity = new double[m_similarity.length];
		while(simDiff > 1e-16){
			if(!sync.train(trainSet))
				m_failCount++;// Train the model.
		
			// Add Logistics regression to predict the similarities.
			int topK = m_users.get(0).getNeighbors().size();
			
			/***We have two different similarity calculation methods, Euclidean distance and cosine similarity.***/
			//Euclidean distance.
			double[][] EucDiffs = calculateEucSims(sync.getAllAs(), sync.getDimension()*2, topK);
			
			//Cosine similarity.
			double[][] CosineSims = calculateCosineSims(sync.getAllAs(), sync.getDimension()*2, topK);
			
			//We can either pass Euclidean distance similarity or cosine similarity.
			LogisticRegression4Similarity lr4Sim = new LogisticRegression4Similarity(m_users, CosineSims);
			
			//Train the weights of each user's models.
			lr4Sim.train();
			
			//Update similarity based on each user's weight.
			updateSimilarity(lr4Sim.getWeights());
			
			//Calculate the difference of similarities.
			simDiff = calcSimDiff(preSimilarity);
			preSimilarity = Arrays.copyOf(m_similarity, m_similarity.length);
			iterCount++;
			if(iterCount % 100 == 0)
				System.out.print(".");
		}
		System.out.format("%d iterations in the training process.", iterCount);
		System.out.format("%d fails in batch optimization.\n", m_failCount);
		sync.test(testSet);
	}
	
	public double calcSimDiff(double[] preSim){
		double diff = 0;
		if(preSim.length != m_similarity.length){
			return diff;
		}
		for(int i=0; i<preSim.length; i++){
			diff += (preSim[i] - m_similarity[i]) * (preSim[i] - m_similarity[i]);
		}
		return diff;
	}
	
	//ws: [w0_u0, w1_u0, w0_u1, w1_u1....]
	public void updateSimilarity(double[] ws){
		int index, dim = 2;
		int[] neighborIndexes;
		double sim = 0, bias = 1, vsum = 0;
		double[] wi;
		for(int i=0; i<m_users.size(); i++){
			neighborIndexes = m_users.get(i).getNeighborIndexes();
			for(int j=0; j<neighborIndexes.length; j++){
				index = neighborIndexes[j];
				//Currently, the feature vector is two-dimension.
				wi = Arrays.copyOfRange(ws, i*dim, (i+1)*dim);
				vsum = wi[0] * bias + wi[1] * Utils.cosine(m_users.get(i).getSparse(), m_users.get(index).getSparse());
				sim = 1/(1 + Math.exp(-vsum)); //logit function.
//				System.out.print(sim + "\t");
				m_similarity[getIndex(i, index)] = sim;
			}
		}
	}
	
	//Dim here is the 2*dim, for both a and b.
	public double[][] calculateEucSims(double[] allAs, int dim, int topK){
		double[][] EucSims = new double[m_users.size()][topK];
		double[] EucSim = new double[topK];
		_User user;
		int index;
		int[] neighborIndexes;
		// Calculate the difference between each user with neighbors, take the inverse to get similarity. ||Ai-Aj||^2
		for(int i=0; i<m_users.size(); i++){
			user = m_users.get(i);
			neighborIndexes = user.getNeighborIndexes();
			for(int j=0; j<neighborIndexes.length; j++){
				index = neighborIndexes[j];
				EucSim[j] = calculateOneEucSim(Arrays.copyOfRange(allAs, i*dim, (i+1)*dim), Arrays.copyOfRange(allAs, index*dim, (index+1)*dim));
			}
			EucSims[i] = Arrays.copyOf(EucSim, topK);
			Arrays.fill(EucSim, 0);
		}
		return EucSims;
	}
	
	public double[][] calculateCosineSims(double[] allAs, int dim, int topK){
		double[][] CosineSims = new double[m_users.size()][topK];
		double[] CosineSim = new double[topK];
		_User user;
		int index;
		int[] neighborIndexes;
		// Calculate the difference between each user with neighbors. ||Ai-Aj||^2
		for(int i=0; i<m_users.size(); i++){
			user = m_users.get(i);
			neighborIndexes = user.getNeighborIndexes();
			for(int j=0; j<neighborIndexes.length; j++){
				index = neighborIndexes[j];
				CosineSim[j] = Utils.cosine(Arrays.copyOfRange(allAs, i*dim, (i+1)*dim), Arrays.copyOfRange(allAs, index*dim, (index+1)*dim));
			}
			CosineSims[i] = Arrays.copyOf(CosineSim, topK);
			Arrays.fill(CosineSim, 0);
		}
		return CosineSims;
	}
	public double calculateOneEucSim(double[] Ai, double[] Aj){
		double EucDis = 0;
		if(Ai.length != Aj.length)
			return EucDis;
		for(int i =0; i<Ai.length; i++)
			EucDis += (Ai[i] - Aj[i]) * (Ai[i] - Aj[i]);
		return 1/EucDis;
	}
	
	//Calculate each user's performance.
	public void calcPerformance(){
//		for(int i=0; i<m_TPall.length; i++)
//			Arrays.fill(m_TPall[i], 0);
		for(int i=0; i<m_avgPRF.length; i++)
			Arrays.fill(m_avgPRF[i], 0);

		CoLinAdapt model;
		for(int i=0; i<m_users.size(); i++){
			model = m_users.get(i).getCoLinAdapt();
			model.m_perfStat.calculatePRF();
//			addOneUserTPTable(model.m_perfStat.getTPTable());
			addOneUserPRF(model.m_perfStat.getOneUserPRF());
		}
		for(int i=0; i<m_avgPRF.length; i++){
			for(int j=0; j<m_avgPRF[0].length; j++){
				m_avgPRF[i][j] /= m_users.size();
			}
		}
//		// Print out the TPTable.
//		for (int i = 0; i < m_TPall.length; i++) {
//			for (int j = 0; j < m_TPall[0].length; j++)
//				System.out.print(m_TPall[i][j] + "\t");
//			System.out.println();
//		}
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
