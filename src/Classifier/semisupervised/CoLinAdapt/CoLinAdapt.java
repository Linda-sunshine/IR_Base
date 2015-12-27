/**
 * 
 */
package Classifier.semisupervised.CoLinAdapt;

import java.util.ArrayList;
import java.util.Arrays;

import Classifier.semisupervised.CoLinAdapt._CoLinAdaptStruct.SimType;
import LBFGS.LBFGS;
import LBFGS.LBFGS.ExceptionWithIflag;
import structures._RankItem;
import structures._User;

/**
 * @author Hongning Wang
 * synchronized CoLinAdapt algorithm
 */
public class CoLinAdapt extends LinAdapt {

	double m_eta3; // Weight for R2.
	int m_topK;
	SimType m_sType;
	
	public CoLinAdapt(int classNo, int featureSize, int topK, String globalModel, String featureGroupMap) {
		super(classNo, featureSize, globalModel, featureGroupMap);
		m_eta3 = 0.5;
		m_topK = topK; // when topK<0, we will use a fully connected graph 
		m_sType = SimType.ST_BoW;
	}

	public void setTradeOffs(double eta1, double eta2, double eta3) {
		super.setTradeOffs(eta1, eta2);
		m_eta3 = eta3;
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
		_CoLinAdaptStruct ui, uj;
		double sim;
		for(int i=0; i<m_userList.size(); i++) {
			ui = (_CoLinAdaptStruct)(m_userList.get(i));
			for(int j=i+1; j<m_userList.size(); j++) {
				uj = (_CoLinAdaptStruct)(m_userList.get(j));
				
				sim = ui.getSimilarity(uj, m_sType);
				
				ui.addNeighbor(j, sim);
				uj.addNeighbor(i, sim);
			}
		}
		System.out.format("[Info]Neighborhood graph based on %s constructed for %d users...\n", m_sType, m_userList.size());
	}
	
	//this will be only called once
	@Override
	protected void initLBFGS(){ 
		int vSize = 2*m_dim*m_userList.size();
		
		m_g = new double[vSize];
		m_diag = new double[vSize];
	}
	
	@Override
	protected double calculateFunctionValue(_LinAdaptStruct ui) {
		double fValue = super.calculateFunctionValue(ui), R2 = 0, diff;
		
		_LinAdaptStruct uj;
		for(_RankItem nit:((_CoLinAdaptStruct)ui).getNeighborhood()) {
			uj = m_userList.get(nit.m_index);
			diff = 0;
			for(int k=0; k<m_dim; k++) {
				diff += (ui.getScaling(k) - uj.getScaling(k)) * (ui.getScaling(k) - uj.getScaling(k));
				diff += (ui.getShifting(k) - uj.getShifting(k)) * (ui.getShifting(k) - uj.getShifting(k));
			}
			R2 += nit.m_value * diff;
		}
		return fValue + m_eta3*R2;
	}
	
	protected void calculateGradients(_LinAdaptStruct user){
		super.calculateGradients(user);
		gradientByR2(user);
	}
	
	//Calculate the gradients for the use in LBFGS.
	protected void gradientByR2(_LinAdaptStruct user){		
		_CoLinAdaptStruct uj, ui = (_CoLinAdaptStruct)user;
		int offset = m_dim*2;
		double coef, dA, dB;
		
		for(_RankItem nit:ui.getNeighborhood()) {
			uj = (_CoLinAdaptStruct)m_userList.get(nit.m_index);
			coef = 2* nit.m_value * m_eta3;
			
			for(int k=0; k<m_dim; k++) {
				dA = coef * (ui.getScaling(k) - uj.getScaling(k));
				dB = coef * (ui.getShifting(k) - uj.getShifting(k));
				
				// update ui's gradient
				m_g[offset*ui.m_id + k] += dA;
				m_g[offset*ui.m_id + k + m_dim] += dB;
				
				// update uj's gradient
				m_g[offset*uj.m_id + k] -= dA;
				m_g[offset*uj.m_id + k + m_dim] -= dB;
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
	public void train(){
		int[] iflag = {0}, iprint = {-1, 3};
		double fValue;
		int vSize = 2*m_dim*m_userList.size();
		
		initLBFGS();
		try{
			do{
				fValue = 0;
				Arrays.fill(m_g, 0); // initialize gradient
				
				// accumulate function values and gradients from each user
				for(_LinAdaptStruct user:m_userList) {
					fValue += calculateFunctionValue(user);
					calculateGradients(user);
				}
				
				System.out.println("Fvalue is " + fValue);	
				gradientTest();
				
				LBFGS.lbfgs(vSize, 4, _CoLinAdaptStruct.getSharedA(), fValue, m_g, false, m_diag, iprint, 1e-4, 1e-10, iflag);//In the training process, A is updated.
			} while(iflag[0] != 0);
		} catch(ExceptionWithIflag e) {
			e.printStackTrace();
		}
		
		for(_LinAdaptStruct user:m_userList)
			setPersonalizedModel(user);
	}
}
