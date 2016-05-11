/**
 * 
 */
package Classifier.supervised.modelAdaptation.CoLinAdapt;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;

import Classifier.supervised.modelAdaptation.CoAdaptStruct;
import Classifier.supervised.modelAdaptation._AdaptStruct;
import Classifier.supervised.modelAdaptation._AdaptStruct.SimType;
import LBFGS.LBFGS;
import LBFGS.LBFGS.ExceptionWithIflag;
import structures._PerformanceStat.TestMode;
import structures._RankItem;
import structures._User;
import utils.Utils;

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
	
	@Override
	public String toString() {
		return String.format("CoLinAdapt[dim:%d,eta1:%.3f,eta2:%.3f,eta3:%.3f,eta4:%.3f,k:%d,NB:%s]", m_dim, m_eta1, m_eta2, m_eta3, m_eta4, m_topK, m_sType);
	}
	
	public void setR2TradeOffs(double eta3, double eta4) {
		m_eta3 = eta3;
		m_eta4 = eta4;
	}
	
	public void setSimilarityType(SimType sType) {
		m_sType = sType;
	}
	
	@Override
	int getVSize() {
		return 2*m_dim*m_userList.size();
	}
	
	void constructUserList(ArrayList<_User> userList) {
		int vSize = 2*m_dim;
		
		//step 1: create space
		m_userList = new ArrayList<_AdaptStruct>();		
		for(int i=0; i<userList.size(); i++) {
			_User user = userList.get(i);
			m_userList.add(new _CoLinAdaptStruct(user, m_dim, i, m_topK));
		}
		m_pWeights = new double[m_gWeights.length];			
		
		//huge space consumption
		_CoLinAdaptStruct.sharedA = new double[getVSize()];
		
		//step 2: copy each user's A to shared A in _CoLinAdaptStruct		
		_CoLinAdaptStruct user;
		for(int i=0; i<m_userList.size(); i++) {
			user = (_CoLinAdaptStruct)m_userList.get(i);
			System.arraycopy(user.m_A, 0, _CoLinAdaptStruct.sharedA, vSize*i, vSize);
		}
	}

	@Override
	public void loadUsers(ArrayList<_User> userList){		
		//step 1: create space
		constructUserList(userList);
		
		//step 2: construct neighborhood graph
		constructNeighborhood(m_sType);

//		// Print out similarity.
//		PrintWriter writer;
//		try {
//			writer = new PrintWriter(new File("constrain_sim.txt"));
//			for(_AdaptStruct u: m_userList){
//				_CoLinAdaptStruct ui = (_CoLinAdaptStruct)u;
//				if(ui.getNeighbors().size() < 200)
//					System.out.println("Smaller than 200!!"+ui.getUserID());
//				for(_RankItem nit: ui.getNeighbors())
//					writer.write(nit.m_value+",");
//				writer.write("\n");
//			}
//			writer.close();
//		} catch (FileNotFoundException e) {
//			e.printStackTrace();
//		}
	}
	
	//this will be only called once in CoLinAdapt
	@Override
	protected void initLBFGS(){ 
		int vSize = getVSize();
		
		m_g = new double[vSize];
		m_diag = new double[vSize];
	}
	
	@Override
	protected double calculateFuncValue(_AdaptStruct u) {		
		double fValue = super.calculateFuncValue(u), R2 = 0, diffA, diffB;
			
		//R2 regularization
		_CoLinAdaptStruct ui = (_CoLinAdaptStruct)u, uj;
		for(_RankItem nit:ui.getNeighbors()) {
			uj = (_CoLinAdaptStruct)m_userList.get(nit.m_index);
			diffA = 0;
			diffB = 0;
			for(int k=0; k<m_dim; k++) {
				diffA += (ui.getScaling(k) - uj.getScaling(k)) * (ui.getScaling(k) - uj.getScaling(k));
				diffB += (ui.getShifting(k) - uj.getShifting(k)) * (ui.getShifting(k) - uj.getShifting(k));
			}
			R2 += nit.m_value * (m_eta3*diffA + m_eta4*diffB);
//			R2 += 0.1 * (m_eta3*diffA + m_eta4*diffB);
//			R2 += (nit.m_value / simSum) * (m_eta3*diffA + m_eta4*diffB);
		}
		return fValue + R2;
	}
	
//	@Override
//	protected double calculateFuncValue(_AdaptStruct u){
//		_CoLinAdaptStruct ui = (_CoLinAdaptStruct)u, uj;
//		
//		double L = calcLogLikelihood(ui); //log likelihood.
//		double R1 = 0, R2 = 0, diffA, diffB;
//		
//		//R1 regularization.
//		for(int i=0; i<ui.getPWeights().length; i++){
//			R1 += m_eta1 * Utils.EuclideanDistance(ui.getPWeights(), m_gWeights);
//		}
//		
//		//R2 regularization
//		for(_RankItem nit:ui.getNeighbors()) {
//			uj = (_CoLinAdaptStruct)m_userList.get(nit.m_index);
//			diffA = 0;
//			diffB = 0;
//			for(int k=0; k<m_dim; k++) {
//				diffA += (ui.getScaling(k) - uj.getScaling(k)) * (ui.getScaling(k) - uj.getScaling(k));
//				diffB += (ui.getShifting(k) - uj.getShifting(k)) * (ui.getShifting(k) - uj.getShifting(k));
//			}
//			R2 += nit.m_value * (m_eta3*diffA + m_eta4*diffB);
////			R2 += 0.1 * (m_eta3*diffA + m_eta4*diffB);
////			R2 += (nit.m_value / simSum) * (m_eta3*diffA + m_eta4*diffB);
//		}
//		return R1 + R2 - L;
//	}
	
	@Override
	protected void calculateGradients(_AdaptStruct u){
		super.calculateGradients(u);
		gradientByR2(u);
	}
	
//	//Calculate the gradients for the use in LBFGS.
//	@Override
//	protected void gradientByR1(_AdaptStruct u){
//		_CoLinAdaptStruct user = (_CoLinAdaptStruct)u;
//		double dA, dB;
//		int k, offset = 2*m_dim*user.getId();//general enough to accommodate both LinAdapt and CoLinAdapt
//		//R1 regularization part
//		for(int n=0; n<m_featureSize+1; n++){
//			k = m_featureGroupMap[n];
//			dA = m_eta1 * (user.getPWeights()[n] - m_gWeights[n]) * m_gWeights[n];
//			dB = m_eta1 * (user.getPWeights()[n] - m_gWeights[n]);
//			
//			m_g[offset + k] += dA;
//			m_g[offset + k] += dB;
//		}
//	}
	
	//Calculate the gradients for the use in LBFGS.
	protected void gradientByR2(_AdaptStruct user){		
		_CoLinAdaptStruct ui = (_CoLinAdaptStruct)user, uj;
		int offseti = m_dim*2*ui.getId(), offsetj;
		double coef, dA, dB;
		
		for(_RankItem nit:ui.getNeighbors()) {
			uj = (_CoLinAdaptStruct)m_userList.get(nit.m_index);
			offsetj = m_dim*2*uj.getId();
			coef = 2 * nit.m_value;
//			coef = 2 * 0.1;
//			coef = 2 * nit.m_value/simSum;

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
		
		if (m_displayLv==2)
			System.out.format("Gradient magnitude for a: %.5f, b: %.5f\n", magA, magB);
		return magA + magB;
	}
	
	//this is batch training in each individual user
	@Override
	public double train(){
		int[] iflag = {0}, iprint = {-1, 3};
		double fValue, oldFValue = Double.MAX_VALUE;;
		int vSize = getVSize(), displayCount = 0;
		double oldMag = 0;
		_LinAdaptStruct user;
		
		initLBFGS();
		init();
		try{
			do{
				fValue = 0;
				Arrays.fill(m_g, 0); // initialize gradient				
//				setPersonalizedModel();
				// accumulate function values and gradients from each user
				for(int i=0; i<m_userList.size(); i++) {
					user = (_LinAdaptStruct)m_userList.get(i);
					fValue += calculateFuncValue(user);
					calculateGradients(user);
				}
				if (m_displayLv==2) {
					gradientTest();
					System.out.println("Fvalue is " + fValue);
				} else if (m_displayLv==1) {
					if (fValue<oldFValue)
						System.out.print("o");
					else
						System.out.print("x");
					
					if (++displayCount%100==0)
						System.out.println();
				} 
				oldFValue = fValue;
				
				LBFGS.lbfgs(vSize, 5, _CoLinAdaptStruct.getSharedA(), fValue, m_g, false, m_diag, iprint, 1e-3, 1e-16, iflag);//In the training process, A is updated.
//				setPersonalizedModel();
			} while(iflag[0] != 0);
			System.out.println();
		} catch(ExceptionWithIflag e) {
			System.out.println("LBFGS fails!!!!");
			e.printStackTrace();
		}		
		
		setPersonalizedModel();
		return oldFValue;
	}
}
