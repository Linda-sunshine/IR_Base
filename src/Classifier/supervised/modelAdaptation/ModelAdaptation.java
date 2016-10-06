/**
 * 
 */
package Classifier.supervised.modelAdaptation;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;

import Classifier.BaseClassifier;
import Classifier.supervised.modelAdaptation._AdaptStruct.SimType;
import structures._Doc;
import structures._PerformanceStat;
import structures._PerformanceStat.TestMode;
import structures._RankItem;
import structures._Review;
import structures._Review.rType;
import structures._User;
import utils.Utils;

/**
 * @author Hongning Wang
 * abstract class for model adaptation algorithms
 */
public abstract class ModelAdaptation extends BaseClassifier {
	protected ArrayList<_AdaptStruct> m_userList; // references to the users	
	protected int m_userSize; // valid user size
	
	protected double[] m_gWeights; //global model weight
	protected double[] m_pWeights; // cache for personalized weight

	protected TestMode m_testmode; // test mode of different algorithms 
	protected int m_displayLv = 1;//0: display nothing during training; 1: display the change of objective function; 2: display everything

	//if we will set the personalized model to the target user (otherwise use the global model)
	protected boolean m_personalized;

	// Decide if we will normalize the likelihood.
	protected boolean m_LNormFlag=true;
	protected String m_dataset = "Amazon"; // Default dataset.

	// added by Lin.
	public ModelAdaptation(int classNo, int featureSize) {
		super(classNo, featureSize);
		m_pWeights = null;
		m_personalized = true;
	}
	
	public ModelAdaptation(int classNo, int featureSize, HashMap<String, Integer> featureMap, String globalModel) {
		super(classNo, featureSize);
		
		loadGlobalModel(featureMap, globalModel);
		m_pWeights = null;
		m_personalized = true;
	}
	
	public void setDisplayLv(int level) {
		m_displayLv = level;
	}
	
	public void setTestMode(TestMode mode) {
		m_testmode = mode;
	}
	
	public void setPersonalization(boolean p) {
		m_personalized = p;
	}	
	
	public void setLNormFlag(boolean b){
		m_LNormFlag = b;
	}
	
	//Load global model from file.
	public void loadGlobalModel(HashMap<String, Integer> featureMap, String filename){
		if (featureMap==null || filename==null)
			return;
		
		try{
			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(filename), "UTF-8"));
			String line, features[];
			int pos;
			
			m_gWeights = new double[m_featureSize+1];//to include the bias term
			while((line=reader.readLine()) != null) {
				features = line.split(":");
				if (features[0].equals("BIAS"))
					m_gWeights[0] = Double.valueOf(features[1]);
				else if (featureMap.containsKey(features[0])){
					pos = featureMap.get(features[0]);
					if (pos>=0 && pos<m_featureSize)
						m_gWeights[pos+1] = Double.valueOf(features[1]);
					else
						System.err.println("[Warning]Unknown feature " + features[0]);
				} else 
					System.err.println("[Warning]Unknown feature " + features[0]);
			}
			
			reader.close();
		} catch(IOException e){
			System.err.format("[Error]Fail to open file %s.\n", filename);
		}
	}
	
	abstract public void loadUsers(ArrayList<_User> userList);
	
	protected void constructNeighborhood(final SimType sType) {
		int numberOfCores = Runtime.getRuntime().availableProcessors();
		ArrayList<Thread> threads = new ArrayList<Thread>();
		
		for(int k=0; k<numberOfCores; ++k){
			threads.add((new Thread() {
				int core, numOfCores;
				
				@Override
				public void run() {
					CoAdaptStruct ui, uj;
					try {
						for (int i = 0; i + core <m_userList.size(); i += numOfCores) {
							ui = (CoAdaptStruct)m_userList.get(i+core);
							for(int j=0; j<m_userList.size(); j++) {
								if (j == i+core)
									continue;
								uj = (CoAdaptStruct)(m_userList.get(j));
								
								ui.addNeighbor(j, ui.getSimilarity(uj, sType));
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

		System.out.format("[Info]Neighborhood graph based on %s constructed for %d users...\n", sType, m_userList.size());
	}
	
	protected int[] constructReverseNeighborhood() {
		int adaptSize = 0;//total number of adaptation instances
		
		//construct the reverse link
		CoAdaptStruct ui, uj;
		for(int i=0; i<m_userList.size(); i++) {
			ui = (CoAdaptStruct)(m_userList.get(i));
			for(_RankItem nit:ui.getNeighbors()) {
				uj = (CoAdaptStruct)(m_userList.get(nit.m_index));//uj is a neighbor of ui
				
				uj.addReverseNeighbor(i, nit.m_value);
			}
			adaptSize += ui.getAdaptationSize();
		}
		
		//construct the order of online updating
		ArrayList<_RankItem> userorder = new ArrayList<_RankItem>();
		for(int i=0; i<m_userList.size(); i++) {
			ui = (CoAdaptStruct)(m_userList.get(i));
			
			for(_Review r:ui.getReviews()) {//reviews in each user is already ordered by time
				if (r.getType() == rType.ADAPTATION) {
					userorder.add(new _RankItem(i, r.getTimeStamp()));//to be in ascending order
				}
			}
		}
		
		Collections.sort(userorder);
		
		int[] userOrder = new int[adaptSize];
		for(int i=0; i<adaptSize; i++)
			userOrder[i] = userorder.get(i).m_index;
		return userOrder;
	}
	
	
	@Override
	protected void init(){
		m_userSize = 0;//need to get the total number of valid users to construct feature vector for MT-SVM
		for(_AdaptStruct user:m_userList){			
			if (user.getAdaptationSize()>0) 				
				m_userSize ++;	
			user.getPerfStat().clear(); // clear accumulate performance statistics
		}
	}
	
	protected int getAdaptationSize(_AdaptStruct user) {
		return user.getAdaptationSize();
	}
	
	abstract protected void setPersonalizedModel();
	
	@Override
	public double test(){
		int numberOfCores = Runtime.getRuntime().availableProcessors();
		ArrayList<Thread> threads = new ArrayList<Thread>();
		
		for(int k=0; k<numberOfCores; ++k){
			threads.add((new Thread() {
				int core, numOfCores;
				public void run() {
					_AdaptStruct user;
					_PerformanceStat userPerfStat;
					try {
						for (int i = 0; i + core <m_userList.size(); i += numOfCores) {
							user = m_userList.get(i+core);
							if ( (m_testmode==TestMode.TM_batch && user.getTestSize()<1) // no testing data
								|| (m_testmode==TestMode.TM_online && user.getAdaptationSize()<1) // no adaptation data
								|| (m_testmode==TestMode.TM_hybrid && user.getAdaptationSize()<1) && user.getTestSize()<1) // no testing and adaptation data 
								continue;
								
							userPerfStat = user.getPerfStat();								
							if (m_testmode==TestMode.TM_batch || m_testmode==TestMode.TM_hybrid) {				
								//record prediction results
								for(_Review r:user.getReviews()) {
									if (r.getType() != rType.TEST)
										continue;
									int trueL = r.getYLabel();
									int predL = user.predict(r); // evoke user's own model
									userPerfStat.addOnePredResult(predL, trueL);
								}
							}							
							userPerfStat.calculatePRF();	
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
		
		int count = 0;
		double[] macroF1 = new double[m_classNo];
		_PerformanceStat userPerfStat;
		
		for(_AdaptStruct user:m_userList) {
			if ( (m_testmode==TestMode.TM_batch && user.getTestSize()<1) // no testing data
				|| (m_testmode==TestMode.TM_online && user.getAdaptationSize()<1) // no adaptation data
				|| (m_testmode==TestMode.TM_hybrid && user.getAdaptationSize()<1) && user.getTestSize()<1) // no testing and adaptation data 
				continue;
			
			userPerfStat = user.getPerfStat();
			for(int i=0; i<m_classNo; i++)
				macroF1[i] += userPerfStat.getF1(i);
			m_microStat.accumulateConfusionMat(userPerfStat);
			count ++;
		}
		
		System.out.println(toString());
		calcMicroPerfStat();
		
		// macro average
		System.out.println("\nMacro F1:");
		for(int i=0; i<m_classNo; i++)
			System.out.format("Class %d: %.4f\t", i, macroF1[i]/count);
		System.out.println("\n");
		return Utils.sumOfArray(macroF1);
	}

	@Override
	public void saveModel(String modelLocation) {	
		for(_AdaptStruct user:m_userList) {
			try {
	            BufferedWriter writer = new BufferedWriter(new FileWriter(modelLocation+"/"+user.getUserID()+".classifer"));
	            StringBuilder buffer = new StringBuilder(512);
	            double[] pWeights = user.getPWeights();
	            for(int i=0; i<pWeights.length; i++) {
	            	buffer.append(pWeights[i]);
	            	if (i<pWeights.length-1)
	            		buffer.append(',');
	            }
	            writer.write(buffer.toString());
	            writer.close();
	        } catch (Exception e) {
	            e.printStackTrace(); 
	        } 
		}
		System.out.format("[Info]Save personalized models to %s.", modelLocation);
	}
	
	@Override
	public double train(Collection<_Doc> trainSet) {
		System.err.println("[Error]train(Collection<_Doc> trainSet) is not implemented in ModelAdaptation family!");
		System.exit(-1);
		return Double.NaN;
	}

	@Override
	public int predict(_Doc doc) {//predict by global model		
		System.err.println("[Error]predict(_Doc doc) is not implemented in ModelAdaptation family!");
		System.exit(-1);
		return Integer.MAX_VALUE;
	}

	@Override
	public double score(_Doc d, int label) {//prediction score by global model
		System.err.println("[Error]score(_Doc d, int label) is not implemented in ModelAdaptation family!");
		System.exit(-1);
		return Double.NaN;
	}

	@Override
	protected void debug(_Doc d) {
		System.err.println("[Error]debug(_Doc d) is not implemented in ModelAdaptation family!");
		System.exit(-1);
	}
}
