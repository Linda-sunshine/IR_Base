/**
 * 
 */
package Classifier.semisupervised.CoLinAdapt;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;

import Classifier.semisupervised.CoLinAdapt._CoLinAdaptStruct.SimType;
import LBFGS.LBFGS;
import LBFGS.LBFGS.ExceptionWithIflag;
import structures._RankItem;
import structures._User;
import structures._PerformanceStat.TestMode;

/**
 * @author Hongning Wang
 * synchronized CoLinAdapt algorithm
 */
public class CoLinAdapt extends LinAdapt {

	double m_eta3; // weight for scaling in R2.
	double m_eta4; // weight for shifting in R2.
	int m_topK;
	SimType m_sType = SimType.ST_BoW;// default neighborhood by BoW
	
	public CoLinAdapt(int classNo, int featureSize, HashMap<String, Integer> featureMap, int topK, String globalModel, String featureGroupMap) {
		super(classNo, featureSize, featureMap, globalModel, featureGroupMap);
		m_eta3 = 0.5;
		m_eta4 = 0.5;
		m_topK = topK; // when topK<0, we will use a fully connected graph 
		
		// the only possible test modes for CoLinAdapt is batch mode
		m_testmode = TestMode.TM_batch;
	}

	public void setR2TradeOffs(double eta3, double eta4) {
		m_eta3 = eta3;
		m_eta4 = eta4;
	}
	
	public void setSimilarityType(SimType sType) {
		m_sType = sType;
	}
	
	@Override
	public void loadUsers(ArrayList<_User> userList){	
		int vSize = 2*m_dim;
	
		//step 1: create space
		m_userList = new ArrayList<_LinAdaptStruct>();		
		for(int i=0; i<userList.size(); i++) {
			_User user = userList.get(i);
			m_userList.add(new _CoLinAdaptStruct(user, m_dim, i, m_topK));
		}
		m_pWeights = new double[m_gWeights.length];			
		
		//huge space consumption
		_CoLinAdaptStruct.sharedA = new double[vSize*m_userList.size()];
		
		//step 2: copy each user's A to shared A in _CoLinAdaptStruct		
		_LinAdaptStruct user;
		for(int i=0; i<m_userList.size(); i++) {
			user = m_userList.get(i);
			System.arraycopy(user.m_A, 0, _CoLinAdaptStruct.sharedA, vSize*i, vSize);
		}
		
		//step 3: construct neighborhood graph
		constructNeighborhood();
	}
	
	void constructNeighborhood() {
		int numberOfCores = Runtime.getRuntime().availableProcessors();
		ArrayList<Thread> threads = new ArrayList<Thread>();
		
		for(int k=0; k<numberOfCores; ++k){
			threads.add((new Thread() {
				int core, numOfCores;
				public void run() {
					_CoLinAdaptStruct ui, uj;
					try {
						for (int i = 0; i + core <m_userList.size(); i += numOfCores) {
							ui = (_CoLinAdaptStruct)(m_userList.get(i+core));
							for(int j=0; j<m_userList.size(); j++) {
								if (j == i+core)
									continue;
								uj = (_CoLinAdaptStruct)(m_userList.get(j));
								
								ui.addNeighbor(j, ui.getSimilarity(uj, m_sType));
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

		System.out.format("[Info]Neighborhood graph based on %s constructed for %d users...\n", m_sType, m_userList.size());
	}
	
	//this will be only called once in CoLinAdapt
	@Override
	protected void initLBFGS(){ 
		int vSize = 2*m_dim*m_userList.size();
		
		m_g = new double[vSize];
		m_diag = new double[vSize];
	}
	
	@Override
	protected double calculateFuncValue(_LinAdaptStruct ui) {
		double fValue = super.calculateFuncValue(ui), R2 = 0, diffA, diffB;
		
		//R2 regularization
		_LinAdaptStruct uj;
		for(_RankItem nit:((_CoLinAdaptStruct)ui).getNeighbors()) {
			uj = m_userList.get(nit.m_index);
			diffA = 0;
			diffB = 0;
			for(int k=0; k<m_dim; k++) {
				diffA += (ui.getScaling(k) - uj.getScaling(k)) * (ui.getScaling(k) - uj.getScaling(k));
				diffB += (ui.getShifting(k) - uj.getShifting(k)) * (ui.getShifting(k) - uj.getShifting(k));
			}
			R2 += nit.m_value * (m_eta3*diffA + m_eta4*diffB);
		}
		return fValue + R2;
	}
	
	@Override
	protected void calculateGradients(_LinAdaptStruct user){
		super.calculateGradients(user);
		gradientByR2(user);
	}
	
	//Calculate the gradients for the use in LBFGS.
	protected void gradientByR2(_LinAdaptStruct user){		
		_CoLinAdaptStruct ui = (_CoLinAdaptStruct)user, uj;
		int offseti = m_dim*2*ui.m_id, offsetj;
		double coef, dA, dB;
		
		for(_RankItem nit:ui.getNeighbors()) {
			uj = (_CoLinAdaptStruct)m_userList.get(nit.m_index);
			offsetj = m_dim*2*uj.m_id;
			coef = 2 * nit.m_value;
			
			for(int k=0; k<m_dim; k++) {
				dA = coef * m_eta3 * (ui.getScaling(k) - uj.getScaling(k));
				dB = coef * m_eta4 * (ui.getShifting(k) - uj.getShifting(k));
				
				// update ui's gradient
				m_g[offseti + k] += dA;
				m_g[offseti + k + m_dim] += dB;
				
				// update uj's gradient
				m_g[offsetj + k] -= dA;
				m_g[offsetj + k + m_dim] -= dB;
			}			
		}
	}
	
	@Override
	protected double gradientTest() {
		int vSize = 2*m_dim, offset, uid;
		double magA = 0, magB = 0;
		for(int n=0; n<m_userList.size(); n++) {
			uid = n*vSize;
			for(int i=0; i<m_dim; i++){
				offset = uid + i;
				magA += m_g[offset]*m_g[offset];
				magB += m_g[offset+m_dim]*m_g[offset+m_dim];
			}
		}
		System.out.format("Gradient magnitude for a: %.5f, b: %.5f\n", magA, magB);
		return magA + magB;
	}
	
	//this is batch training in each individual user
	@Override
	public double train(){
		int[] iflag = {0}, iprint = {-1, 3};
		double fValue, oldFValue = Double.MAX_VALUE;;
		int vSize = 2*m_dim*m_userList.size(), displayCount = 0;
		
		initLBFGS();
		init();
		try{
			do{
				fValue = 0;
				Arrays.fill(m_g, 0); // initialize gradient				
				
				// accumulate function values and gradients from each user
				for(_LinAdaptStruct user:m_userList) {
					fValue += calculateFuncValue(user);
					calculateGradients(user);
				}
				
				if (m_displayLv==2) {
					System.out.println("Fvalue is " + fValue);
					gradientTest();
				} else if (m_displayLv==1) {
					if (fValue<oldFValue)
						System.out.print("o");
					else
						System.out.print("x");
					
					if (++displayCount%100==0)
						System.out.println();
				} 
				oldFValue = fValue;
				
				LBFGS.lbfgs(vSize, 5, _CoLinAdaptStruct.getSharedA(), fValue, m_g, false, m_diag, iprint, 5e-2, 1e-16, iflag);//In the training process, A is updated.
			} while(iflag[0] != 0);
			System.out.println();
		} catch(ExceptionWithIflag e) {
			e.printStackTrace();
		}		
		
		for(_LinAdaptStruct user:m_userList)
			setPersonalizedModel(user);
		return oldFValue;
	}
}
