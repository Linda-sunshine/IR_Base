//package Application;
package Application;
import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;

import structures.Pair;

public class PostProcess {
	HashMap<String, ArrayList<String>> m_userPairMap;
	ArrayList<ArrayList<String>> m_users;
	double[] m_NDCGs, m_MAPs;
	Object m_NDCGMAPLock = new Object();
	
	public PostProcess(){
		m_userPairMap = new HashMap<String, ArrayList<String>>();
	}
	
	public void loadData(String filename){
		if (filename==null || filename.isEmpty())
			return;
		
		try {
			int pairCount = 0;
			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(filename), "UTF-8"));
			String line;
			while ((line = reader.readLine()) != null) {
				int index = line.indexOf(',');
				String userID = line.substring(0, index);
				if(!m_userPairMap.containsKey(userID)){
					m_userPairMap.put(userID, new ArrayList<String>());
				}
				m_userPairMap.get(userID).add(line.substring(index+1));
				pairCount++;
			}
			reader.close();
			System.out.format("(%d, %d) user/pairs are loaded from %s...\n", m_userPairMap.size(), pairCount, filename);
			
		} catch (IOException e) {
			System.err.format("[Error]Failed to open file %s!!", filename);
		}
	}
	
	// The function for calculating all NDCGs and MAPs.
	public void calculateAllNDCGMAP(){
		m_NDCGs = new double[m_userPairMap.size()];
		m_MAPs = new double[m_userPairMap.size()];
		
		System.out.print("[Info]Start calculating NDCG and MAP...\n");
		int numberOfCores = Runtime.getRuntime().availableProcessors();
		ArrayList<Thread> threads = new ArrayList<Thread>();
		m_users = new ArrayList<ArrayList<String>>();
		
		for(String uid: m_userPairMap.keySet()){
			m_users.add(m_userPairMap.get(uid));
		}
		for(int k=0; k<numberOfCores; ++k){
			threads.add((new Thread() {
				int core, numOfCores;
				@Override
				public void run() {
					ArrayList<String> u;
					try {
						for (int i = 0; i + core <m_users.size(); i += numOfCores) {
							if(i%500==0) System.out.print(".");
							u = m_users.get(i+core);
							double[] vals = calculateNDCGMAP(u);
							// put the calculated nDCG into the array for average calculation
							synchronized(m_NDCGMAPLock){
								m_NDCGs[i+core] = vals[0];
								m_MAPs[i+core] = vals[1];
							}
						}
					} catch(Exception ex) {
							ex.printStackTrace(); 
					}
				}
					
				private Thread initialize(int core, int numOfCores) {
					this.core = core;
					this.numOfCores = numOfCores;
					return this;
				}
			}).initialize(k, numberOfCores));
			threads.get(k).start();
		}
			
		for(int k=0;k<numberOfCores;++k){
			try {
				threads.get(k).join();
			} catch (InterruptedException e) {
				e.printStackTrace();
			} 
		}
	}
	
	public void calculateAvgNDCGMAP(){
		double avgNDCG = 0, avgMAP = 0;
		for(int i=0; i<m_NDCGs.length; i++){
			avgNDCG += m_NDCGs[i];
			avgMAP += m_MAPs[i];
		}
		avgNDCG /= m_NDCGs.length;
		avgMAP /= m_MAPs.length;
		System.out.format("\n[Info]Avg NDCG, MAP -- %.5f\t%.5f\n\n", avgNDCG, avgMAP);
	}
	
	public void reverse(Pair[] rank){
		int p1 = 0, p2 = rank.length-1;
		while(p1 < p2){
			Pair tmp = rank[p2];
			rank[p2] = rank[p1];
			rank[p1] = tmp;
			p1++;
			p2--;
		}
	}
	// calculate the nDCG and MAP for each user
	public double[] calculateNDCGMAP(ArrayList<String> pairs){
		double iDCG = 0, DCG = 0, PatK = 0, AP = 0, count = 0;
			
		int[] rank = new int[pairs.size()];
		Pair[] realRank = new Pair[pairs.size()];
		
		//Calculate the ideal rank and real rank.
		for(int i=0; i<pairs.size(); i++){
			String[] strs = pairs.get(i).split(",");
			if(strs.length != 3){
				System.out.println(pairs.get(i));
				continue;
			}
			rank[i] = Integer.valueOf(strs[1]);
			realRank[i] = new Pair(rank[i], Double.valueOf(strs[2]));
		}
		
		// sort the array in descending order
		sortPrimitivesDescending(rank);
		reverse(realRank);
		// sort the calculated rank based on each pair's value
		// sort the calculated rank based on each pair's value
		Arrays.sort(realRank, new Comparator<Pair>(){
			@Override
			public int compare(Pair p1, Pair p2){
				if(p1.getValue() < p2.getValue())
					return 1;
				else if(p1.getValue() > p2.getValue())
					return -1;
				else{
//					return 0;
					return (int) (-p1.getLabel() + p2.getLabel());
				}
			}
		});					
		//Calculate DCG and iDCG, nDCG = DCG/iDCG.
		for(int i=0; i<rank.length; i++){
			iDCG += (Math.pow(2, rank[i])-1)/(Math.log(i+2));//log(i+1), since i starts from 0, add 1 more.
			DCG += (Math.pow(2, realRank[i].getLabel())-1)/(Math.log(i+2));
			if(realRank[i].getLabel() >= 1){
				PatK = (count+1)/((double)i+1);
				AP += PatK;
				count++;
			}
		}
		return new double[]{DCG/iDCG, AP/count};
	}
	
	// load the true label and predicted label of testing pairs after svd
	public void loadTruePredFiles(String testFile, String predFile){
		System.out.println("Start loading true label and pred score....");
		if(testFile == null || predFile == null)
			return;
		try {
			// load the testFile first
			BufferedReader testReader = new BufferedReader(new InputStreamReader(new FileInputStream(testFile), "UTF-8"));
			BufferedReader predReader = new BufferedReader(new InputStreamReader(new FileInputStream(predFile), "UTF-8"));
			String testLine, predLine;
			// skip the first three lines 
			int skip = 0;
			while(skip++ < 3){
				testReader.readLine();
				predReader.readLine();
			}
			while ((testLine = testReader.readLine()) != null &&(predLine = predReader.readLine()) != null) {
				String[] testStrs = testLine.split("\\s+");
				String[] predStrs = predLine.split("\\s+");
				String userID = testStrs[0];
				String pair = String.format("%s,%s,%s\n", testStrs[1], testStrs[2], predStrs[2]);
				if(!m_userPairMap.containsKey(userID)){
					m_userPairMap.put(userID, new ArrayList<String>());
				}
				m_userPairMap.get(userID).add(pair);
			}
			testReader.close();
			predReader.close();
			System.out.format("%d users' testing results are loaded.\n", m_userPairMap.size());
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	public void sortPrimitivesDescending(int[] rank){
		Arrays.sort(rank);
		// then reverse the array
		for(int i=0; i<rank.length/2; i++){
			int tmp = rank[rank.length-1-i];
			rank[rank.length-i-1] = rank[i];
			rank[i] = tmp;
		}
	}
	
	public static void main(String[] args){
		for(int d: new int[]{5, 10}){
			for(String dataset: new String[]{"YelpNew"}){
				for(int t: new int[]{6}){
		String model = "svd";
		String setting = "all_nei"; // "all_nei"
		
		PostProcess process = new PostProcess();
		if(model.equals("fm")){
			System.out.format("-----------%s, order:%d, d:%d------------\n", dataset, t, d);
			String predFile = String.format("./data/linkPredData/link_d_%d/%s_link_pred_order_%d_d_%d_fm_ials.txt", d, dataset, t, d);
			process.loadData(predFile);
			process.calculateAllNDCGMAP();
			process.calculateAvgNDCGMAP();
		} else if(model.equals("svd")){
			System.out.format("-----------%s, order:%d, d:%d------------\n", dataset, t, d);
			String testMMFile = String.format("./data/linkPredData/svd/%s_link_pred_order_%d_test.mm", dataset, t);
			String predFile = String.format("./data/linkPredData/link_d_%d/%s_link_pred_order_%d_test.mm.predict", d, dataset, t);
			process.loadTruePredFiles(testMMFile, predFile);
			process.calculateAllNDCGMAP();
			process.calculateAvgNDCGMAP();
		}}}}}
//		
//		// cf post processing
//		if(setting.equals("topk")){
//			for(int t: new int[]{2}){
//				for(int k: new int[]{4}){
//					PostProcess process = new PostProcess();
//					if(model.equals("fm")){
//						String predFile = String.format("./data/cfData/%s_cf_time_%d_topk_%d_d_%d_fm_text.txt", dataset, t, k, d);
//						process.loadData(predFile);
//						System.out.format("-----time-%d--topk--%d----\n", t, k);
//						process.calculateAllNDCGMAP();
//						process.calculateAvgNDCGMAP();
//					} else if(model.equals("svd")){
//						String testMMFile = String.format("./data/cfData/svd/%s_cf_time_%d_topk_%d_test.mm", dataset, t, k);
//						String predFile = String.format("./data/cfData/svd_%s_%d/%s_cf_time_%d_topk_%d_test.mm.predict", setting, d, dataset, t, k);
//						process.loadTruePredFiles(testMMFile, predFile);
//						System.out.format("-----time-%d--topk--%d----\n", t, k);
//						process.calculateAllNDCGMAP();
//						process.calculateAvgNDCGMAP();
//					}
//				}
//			}
//		} else if(setting.equals("all_nei")){
//			for(int p: new int[]{3, 5, 10}){
//				PostProcess process = new PostProcess();
//				if(model.equals("fm")){
//					String predFile = String.format("./data/cfData/fm_%s_%d/%s_cf_all_nei_pop_%d_d_%d_fm_ials.txt", setting, d, dataset, p, d);
//					process.loadData(predFile);
//					System.out.format("----all nei--pop--%d----\n", p);
//					process.calculateAllNDCGMAP();
//					process.calculateAvgNDCGMAP();
//				} else if(model.equals("svd")){
//					String testMMFile = String.format("./data/cfData/svd/%s_cf_all_nei_pop_%d_test.mm", dataset, p);
//					String predFile = String.format("./data/cfData/svd_%s_%d/%s_cf_all_nei_pop_%d_test.mm.predict", setting, d, dataset, p);
//					process.loadTruePredFiles(testMMFile, predFile);
//					System.out.format("----all nei--pop--%d----\n", p);
//					process.calculateAllNDCGMAP();
//					process.calculateAvgNDCGMAP();
//				}
//			}
//		}
//	}}
}
