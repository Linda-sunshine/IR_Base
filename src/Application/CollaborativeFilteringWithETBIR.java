package Application;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;

import structures.MyPriorityQueue;
import structures._RankItem;
import structures._User;
import utils.Utils;

public class CollaborativeFilteringWithETBIR extends CollaborativeFiltering {
	int m_dim;  // the number of dimension for low-dimension representation
	
	public CollaborativeFilteringWithETBIR(ArrayList<_User> users, int fs, int k) {
		super(users, fs);
		m_dim = k;
	}
	
	// load the P of all the users at once
	@Override
	public void loadUserWeights(String filename, String model, String suffix1, String suffix2){
		m_userWeights = new double[m_users.size()][m_dim*m_dim];
		
		try{
			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(filename), "UTF-8"));
			int userIndex = 0;
			String line, p[];
			while((line = reader.readLine()) != null){
				String userID = line.split("\\s+")[2]; // read user ID
				
				// skip the user without analysis, m_dim * 2 lines
				if(!m_userIDIndex.containsKey(userID)){
					for(int d = 0; d < m_dim; d++){
						reader.readLine();
						reader.readLine();
					}
				} else{
					// find out the user index
					userIndex = m_userIDIndex.get(userID);
				
					// read the p value, dim * dim
					for(int d = 0; d < m_dim; d++){
						reader.readLine();
						p = reader.readLine().split("\\s+");
						if(p.length == m_dim){
							for(int i=0; i<m_dim; i++){
								m_userWeights[userIndex][m_dim*d + i] = Double.valueOf(p[i]);
							}
						}
					}
				}
			}
			reader.close();
			System.out.format("[Info]Finish loading %d users' weights!\n", m_userWeights.length);
		} catch(IOException e){
			System.err.format("[Error]Failed to open file %s!!", filename);
			e.printStackTrace();
		}
	}	
	
	
	// calculate the ranking score of each item of each user
	@Override
	public double calculateRankScore(_User u, String item){
		int userIndex = m_userIDIndex.get(u.getUserID());
		double rankSum = 0;
		double simSum = 0;
			
		if(!m_trainMap.containsKey(item)){
			return 0;
		}
		//select top k users who have purchased this item.
		ArrayList<String> neighbors = m_trainMap.get(item);
		if(m_avgFlag){
			for(String nei: neighbors){
				int neiIndex = m_userIDIndex.get(nei);
				if(neiIndex == userIndex) continue;
				double label = m_users.get(neiIndex).getItemRating(item)+1;
				rankSum += label;
				simSum++;
			}
			if(simSum == 0){
				return 0;
			} else
				return rankSum/simSum;
		} else{
			MyPriorityQueue<_RankItem> topKNeighbors;
			if(neighbors.size() < m_k)
				topKNeighbors = new MyPriorityQueue<_RankItem>(neighbors.size());
			else
				topKNeighbors = new MyPriorityQueue<_RankItem>(m_k);
			//collect k nearest neighbors for each item of the user.
			for(String nei: neighbors){
				int neiIndex = m_userIDIndex.get(nei);
				if(neiIndex == userIndex) continue;
				topKNeighbors.add(new _RankItem(neiIndex, calculateUserItemSimilarity(m_userWeights[userIndex], m_userWeights[neiIndex])));
			}
			//Calculate the value given by the neighbors and similarity;
			for(_RankItem ri: topKNeighbors){
				int label = m_users.get(ri.m_index).getItemRating(item)+1;
				rankSum += m_equalWeight ? label:ri.m_value*label;//If equal weight, add label, otherwise, add weighted label.
				simSum += m_equalWeight ? 1: ri.m_value;
			}
		}
		if(simSum == 0){
			return 0;
		} else
			return rankSum/simSum;
	}
	
	// calculate the similarity between two users based on their P_i^T \gamma
	protected double calculateUserItemSimilarity(double[] ui, double[] item, double[] uj){
		double[] p1Gamma = matrixVectorProduct(ui, item);
		double[] p2Gamma = matrixVectorProduct(uj, item);
		return Utils.cosine(p1Gamma, p2Gamma);
	}	
	
	// P * item
	protected double[] matrixVectorProduct(double[] ui, double[] item){
		
	}
	
	
}
