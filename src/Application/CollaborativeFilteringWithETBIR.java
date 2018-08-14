package Application;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;

import structures.MyPriorityQueue;
import structures._RankItem;
import structures._User;
import utils.Utils;

import javax.rmi.CORBA.Util;

public class CollaborativeFilteringWithETBIR extends CollaborativeFiltering {
	int m_dim;  // the number of dimension for low-dimension representation
	double[][] m_itemWeights;
	HashMap<String, double[]> m_docWeights; //store phi for each doc: key: userIndex_itemIndex; value: m_dim dimension vector
	String m_mode;
	String m_model;
	
	public CollaborativeFilteringWithETBIR(ArrayList<_User> users, int fs, int k, int dim) {
		super(users, fs, k);
		m_dim = dim;
	}

	public void setMode(String mode){ m_mode=mode;}

	public void setModel(String model){m_model = model;}

	public void loadReviewTopicWeights(String filename){
		try{
			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(filename), "UTF-8"));
			m_docWeights = new HashMap<>();
			String line;
			String userID, itemID;
			int userIndex, itemIndex;
			String p[];
			while((line = reader.readLine()) != null){
				//read doc weight (format: No. 0 Doc(user: T5KBc5QbwZ-Oj9ApE4vZJA, item: P7pxQFqr7yBKMMI2J51udw)...)
				userID = line.split("\\s+")[3].replace(",","");
				itemID = line.split("\\s+")[5].replace(")","");

				if(!m_userIDIndex.containsKey(userID) || !m_itemIDIndex.containsKey(itemID)){
					reader.readLine();
				}else{
					//find out the index of user and item
					userIndex = m_userIDIndex.get(userID);
					itemIndex = m_itemIDIndex.get(itemID);

					// read the p value, dim
                    p = reader.readLine().split("\\s+");
                    if (p.length == m_dim) {
                        double[] temp_weight = new double[m_dim];
                        for (int i = 0; i < m_dim; i++) {
                            temp_weight[i] = Double.valueOf(p[i]);
                        }
                        m_docWeights.put(String.format("%d_%d", userIndex, itemIndex), temp_weight);
                    }
				}
			}
			reader.close();
			System.out.format("[Info]Finish loading %d docs' weights.\n",m_docWeights.size());
		} catch(IOException e){
			System.err.format("[Error]Failed to open file %s!!", filename);
			e.printStackTrace();
		}
	}
	
	// load the P of all the users at once
	@Override
	public void loadUserWeights(String filename, String model, String suffix1, String suffix2){
		try{
			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(filename), "UTF-8"));
			int userIndex = 0;
			String line;
			if(model.contains("ETBIR") && (m_mode.contains("Embed") || m_mode.contains("Product"))) {
				m_userWeights = new double[m_users.size()][m_dim*m_dim];
				String p[];
				while ((line = reader.readLine()) != null) {
					String userID = line.split("\\s+")[3]; // read user ID (format: No. x UserID xxx)

					// skip the user without analysis, m_dim * 2 lines
					if (!m_userIDIndex.containsKey(userID)) {
						for (int d = 0; d < m_dim; d++) {
							reader.readLine();
							reader.readLine();
						}
					} else {
						// find out the user index
						userIndex = m_userIDIndex.get(userID);

						// read the p value, dim * dim
						for (int d = 0; d < m_dim; d++) {
							reader.readLine();
							p = reader.readLine().split("\\s+");
							if (p.length == m_dim) {
								for (int i = 0; i < m_dim; i++) {
									m_userWeights[userIndex][m_dim * d + i] = Double.valueOf(p[i]);
								}
							}
						}
					}
				}
			} else {
				m_userWeights = new double[m_users.size()][m_dim];
				while ((line = reader.readLine()) != null) {
					String userID = line.split("[\\(|\\)|\\s]+")[1]; // read user ID (format: ID xxx(30 reviews))

					// skip the user without analysis, m_dim * 2 lines
					if (!m_userIDIndex.containsKey(userID)) {
						for (int d = 0; d < m_dim; d++) {
							reader.readLine();
						}
					} else {
						// find out the user index
						userIndex = m_userIDIndex.get(userID);

						// read the p value, dim * dim
						for (int d = 0; d < m_dim; d++) {
							String p = reader.readLine().split("[\\(|\\)]+")[1];// read weight (format: -- Topic 0(0.03468):	...)
							m_userWeights[userIndex][d] = Double.valueOf(p);
						}
					}
				}
			}
			reader.close();
			System.out.format("[Info]Finish loading %d users' weights.\n", m_userWeights.length);
		} catch(IOException e){
			System.err.format("[Error]Failed to open file %s!!", filename);
			e.printStackTrace();
		}
	}	
	
	// load the learned traits of items 
	public void loadItemWeights(String filename, String model){
		m_itemWeights = new double[m_itemMap.size()][m_dim];
		m_itemIDIndex = new HashMap<String, Integer>();
		m_items = new ArrayList<>();
		
		try{
			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(filename), "UTF-8"));
			String line, itemID;
			String[] eta;
			if(model.contains("ETBIR") && (m_mode.contains("Embed") || m_mode.contains("Product"))){
				while ((line = reader.readLine()) != null) {
//					itemIndex = Integer.valueOf(line.split("\\s+")[1]); // read item index
					itemID = line.split("\\s+")[2]; // read item ID

					// skip the item without analysis, 1 lines
					if (!m_itemMap.containsKey(itemID)) {
						reader.readLine();
						continue;
					} else {
						m_itemIDIndex.put(itemID, m_items.size());
						m_items.add(m_itemMap.get(itemID));
						// read the eta of each item
						eta = reader.readLine().split("\\s+");
						if (eta.length == m_dim) {
							for (int i = 0; i < m_dim; i++) {
								m_itemWeights[m_itemIDIndex.get(itemID)][i] = Double.valueOf(eta[i]);
							}
							m_itemMap.get(itemID).setItemWeights(m_itemWeights[m_itemIDIndex.get(itemID)]);
						}
					}
				}
			} else{
				while ((line = reader.readLine()) != null) {
					itemID = line.split("[\\(|\\)|\\s]+")[1]; // // read item ID (format: ID xxx(30 reviews))
					// skip the item without analysis, m_dim lines
					if (!m_itemMap.containsKey(itemID)) {
						for (int d = 0; d < m_dim; d++) {
							reader.readLine();
						}
						continue;
					} else {
						m_itemIDIndex.put(itemID, m_items.size());
						m_items.add(m_itemMap.get(itemID));
						// read the eta of each item
						// read the p value, dim * dim
						for (int d = 0; d < m_dim; d++) {
							String p = reader.readLine().split("[\\(|\\)]+")[1];// read weight (format: -- Topic 0(0.03468):	...)
							m_itemWeights[m_itemIDIndex.get(itemID)][d] = Double.valueOf(p);
						}
						m_itemMap.get(itemID).setItemWeights(m_itemWeights[m_itemIDIndex.get(itemID)]);
					}
				}
			}
			reader.close();
			System.out.format("[Info]Finish loading %d items' weights!\n", m_itemWeights.length);
		} catch(IOException e){
			System.err.format("[Error]Failed to open file %s!!", filename);
			e.printStackTrace();
		}
	}
	
	
	// calculate the ranking score of each item of each user
	@Override
	public double calculateRankScore(_User u, String item){
		int userIndex = m_userIDIndex.get(u.getUserID());
		int itemIndex = m_itemIDIndex.get(item);
		double rankSum = 0;
		double simSum = 0;
			
		if(!m_trainMap.containsKey(item)){
			return 0;
		}
		//select top k users who have purchased this item.
		ArrayList<String> neighbors;
		if(m_mode.contains("column") || m_mode.equals("userEmbed"))
			neighbors = m_trainMap.get(item); // ID of column users
		else {
			neighbors = u.getTrainItems(); //ID of row items
		}

		if(m_avgFlag){
			if(m_mode.contains("column") || m_mode.equals("userEmbed")) {
				for (String nei : neighbors) {//column users
					int neiIndex = m_userIDIndex.get(nei);
					if (neiIndex == userIndex) continue;
					double label = m_users.get(neiIndex).getItemRating(item) + 1;
					rankSum += label;
					simSum++;
				}
			} else{
				for (String nei : neighbors) {//row items
					double label = u.getItemRating(nei) + 1;
					rankSum += label;
					simSum++;
				}
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
            double[] p1Gamma, p2Gamma;
			//collect k nearest neighbors for each item of the user.
			for(String nei: neighbors){
				int neiIndex;
				if(m_mode.equals("columnPhi") || m_mode.equals("columnPost")){
					/*
					across column users
					columnPhi: compare \phi of doc(userIndex_itemIndex) and \phi of user
					columnPost: compare posterior parameter (\gamma for LDA, softmax(\mu) for CTM and ETBIR)
					 */
					neiIndex = m_userIDIndex.get(nei);
					if(neiIndex == userIndex) continue;
					topKNeighbors.add(new _RankItem(neiIndex,
							Utils.cosine(m_docWeights.get(String.format("%d_%d", neiIndex, itemIndex)), m_userWeights[userIndex])));
				} else if(m_mode.equals("columnProduct")) {
				    /*
				    ETBIR: across column users
				    columnProduct: compare the inner product of user's P and item's \eta
				     */
					neiIndex = m_userIDIndex.get(nei);
					if (neiIndex == userIndex) continue;
                    p1Gamma = matrixVectorProduct(m_userWeights[userIndex], m_itemMap.get(item).getItemWeights());
                    p2Gamma = matrixVectorProduct(m_userWeights[neiIndex], m_itemMap.get(item).getItemWeights());
					topKNeighbors.add(new _RankItem(neiIndex,
                            Utils.cosine(p1Gamma, p2Gamma)));
				} else if(m_mode.equals("userEmbed")){
				    /*
				    across column users
				    userEmbed: P for ETBIR, average over documents across users for LDA (\gamma) and CTM (softmax(\mu))
				     */
					neiIndex = m_userIDIndex.get(nei);
					if (neiIndex == userIndex) continue;
					topKNeighbors.add(new _RankItem(neiIndex, getSimilarity(userIndex, neiIndex)));
				} else if(m_mode.equals("rowPhi") || m_mode.equals("rowPost")){
				    /*
					across row items
					rowPhi: compare \phi of doc(userIndex_itemIndex) and \phi of item
					rowPost: compare posterior parameter (\gamma for LDA, softmax(\mu) for CTM and ETBIR)
					 */
                    neiIndex = m_itemIDIndex.get(nei);
                    if(neiIndex == itemIndex) continue;
                    topKNeighbors.add(new _RankItem(nei,//store item ID rather than index
                            Utils.cosine(m_docWeights.get(String.format("%d_%d", userIndex, neiIndex)), m_itemWeights[itemIndex])));
				} else if(m_mode.equals("rowProduct")){
				    /*
				    across row items
				    rowProduct: compare the inner product of user's P and item's \eta
				     */
                    neiIndex = m_itemIDIndex.get(nei);
                    if (neiIndex == itemIndex) continue;
                    p1Gamma = matrixVectorProduct(m_userWeights[userIndex], m_itemWeights[itemIndex]);
                    p2Gamma = matrixVectorProduct(m_userWeights[userIndex], m_itemWeights[neiIndex]);
                    topKNeighbors.add(new _RankItem(nei,
                            Utils.cosine(p1Gamma, p2Gamma)));
				} else if(m_mode.equals("itemEmbed")){
				    /*
				    across row items
				    itemEmbed: \eta for ETBIR, average over documents across items for LDA (\gamma) and CTM (softmax(\mu))
				     */
                    neiIndex = m_itemIDIndex.get(nei);
                    if (neiIndex == itemIndex) continue;
                    topKNeighbors.add(new _RankItem(nei, Utils.cosine(m_itemWeights[itemIndex], m_itemWeights[neiIndex])));
				}
			}
			//Calculate the value given by the neighbors and similarity;
            int label;
			for(_RankItem ri: topKNeighbors){
			    if(m_mode.equals("columnPhi") || m_mode.equals("columnPost") || m_mode.equals("columnProduct") || m_mode.equals("userEmbed"))
				    label = m_users.get(ri.m_index).getItemRating(item)+1;//ri.index is user's
			    else //ri.index is item's
			        label = u.getItemRating(ri.m_name)+1;
				rankSum += m_equalWeight ? label:ri.m_value*label;//If equal weight, add label, otherwise, add weighted label.
				simSum += m_equalWeight ? 1: ri.m_value;
			}
		}
		if(simSum == 0){
			return 0;
		} else
			return rankSum/simSum;
	}
	
	// P * item_traits, aggregate P in one dimension
	protected double[] matrixVectorProduct(double[] ui, double[] item){
		int dim = item.length;
		if(ui.length % dim != 0){
			System.err.println("[error] Wrong dimension!");
			return null;
		}
		double[] res = new double[dim];
		double sum = 0;
		for(int i=0; i<ui.length; i++){
			sum += ui[i] * item[i % dim];
			if(i % dim == 0){
				res[i / dim] = sum;
				sum = 0;
			}
		}
		return res;
	}
	
	public static void main(String[] args){
		String file = "d/fe/jfos_10_30.txt";
		String[] tokens = file.split("\\.|\\_");
		int dim = Integer.valueOf(tokens[tokens.length-2]);
		System.out.println(dim);
	}
}
