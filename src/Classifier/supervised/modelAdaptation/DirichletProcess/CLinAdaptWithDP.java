package Classifier.supervised.modelAdaptation.DirichletProcess;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.HashMap;

import structures._Doc;
import structures._PerformanceStat;
import structures._Review;
import structures._SparseFeature;
import structures._User;
import structures._thetaStar;
import structures._PerformanceStat.TestMode;
import structures._Review.rType;
import utils.Utils;
import Classifier.supervised.modelAdaptation._AdaptStruct;
/***
 * In this class, we utilize DP to do automatic grouping.
 * The method does not involve global learning.
 * @author lin
 */
public class CLinAdaptWithDP extends CLRWithDP {

	protected double[] m_abNuB = new double[]{1, 0.2}; // prior for scaling
	public static double[] m_supWeights; // newly learned global model, dummy variable in CLinAdaptWithDP
	
	public CLinAdaptWithDP(int classNo, int featureSize,
			HashMap<String, Integer> featureMap, String globalModel, String featureGroupMap) {
		super(classNo, featureSize, featureMap, globalModel);
		loadFeatureGroupMap(featureGroupMap);
		_DPAdaptStruct.m_featureGroupMap = m_featureGroupMap;//this is really an ugly solution
		m_supWeights = m_gWeights;// this design is for evaluate purpose since we don't need to rewrite evaluate.
	}
	
	@Override
	protected void accumulateClusterModels(){
		m_models = new double[getVSize()];
		for(int i=0; i<m_kBar; i++)
			System.arraycopy(m_thetaStars[i].getModel(), 0, m_models, m_dim*2*i, m_dim*2);
	}
	
	@Override
	protected int getVSize() {
		return m_kBar*m_dim*2;// we have global here.
	}

	protected void initPriorG0() {
		m_G0 = new DoubleNormalPrior(m_abNuB[0], m_abNuB[1], m_abNuA[0], m_abNuA[1]);
	}
	
	@Override
	// R1 over each cluster, R1 over super cluster.
	protected double calculateR1(){
		double R1 = 0;
		// Clusters.
		for(int i=0; i<m_kBar; i++)
			R1 += m_G0.logLikelihood(m_thetaStars[i].getModel(), m_eta1, m_eta2);
		
		int offset;
		for(int i=0; i<m_kBar; i++){
			offset = m_dim*2*i;
			for(int k=0; k<m_dim;k++){
				m_g[offset+k] += m_eta1 * (m_models[offset+k]-m_abNuB[0])/m_abNuB[1]/m_abNuB[1]; //scaling
				m_g[offset+k+m_dim] += m_eta2 * (m_models[offset+k+m_dim]-m_abNuA[0])/m_abNuA[1]/m_abNuA[1]; // shifting
			}
		}
		return R1;
	}
	
	@Override
	protected void gradientByFunc(_AdaptStruct u, _Doc review, double weight, double[] g) {
		_DPAdaptStruct user = (_DPAdaptStruct)u;
		
		int n, k; // feature index
		int cIndex = user.getThetaStar().getIndex();
		if(cIndex <0 || cIndex >= m_kBar)
			System.err.println("Error,cannot find the theta star!");
		int offset = m_dim*2*cIndex;
		
		double delta = (review.getYLabel() - logit(review.getSparse(), user)) * weight;
		if(m_LNormFlag)
			delta /= getAdaptationSize(user);
		
		// Bias term for individual user.
		g[offset] -= delta*m_gWeights[0]; //a[0] = ws0*x0; x0=1
		g[offset + m_dim] -= delta;//b[0]
		
		//Traverse all the feature dimension to calculate the gradient for both individual users and super user.
		for(_SparseFeature fv: review.getSparse()){
			n = fv.getIndex() + 1;
			k = m_featureGroupMap[n];
			g[offset + k] -= delta*m_gWeights[n]*fv.getValue(); // w_si*x_di
			g[offset + m_dim + k] -= delta*fv.getValue(); // x_di
		}
	}
	
	@Override
	public void loadUsers(ArrayList<_User> userList) {
		m_userList = new ArrayList<_AdaptStruct>();
		// Init each user.
		for(_User user:userList)
			m_userList.add(new _DPAdaptStruct(user, m_dim));

		m_pWeights = new double[m_gWeights.length];		
	}
	
	// Logit function is different from the father class.
	@Override
	protected double logit(_SparseFeature[] fvs, _AdaptStruct u){
		int k, n;
		_DPAdaptStruct user = (_DPAdaptStruct)u;
		double[] Au = user.getThetaStar().getModel();
		double value = Au[0]*m_gWeights[0] + Au[m_dim];//Bias term: w_s0*a0+b0.
		for(_SparseFeature fv: fvs){
			n = fv.getIndex() + 1;
			k = m_featureGroupMap[n];
			value += (Au[k]*m_gWeights[n] + Au[m_dim+k]) * fv.getValue();
		}
		return Utils.logistic(value);
	}
	
	@Override
	protected void setPersonalizedModel() {
		double[] As;
		int ki;
		_DPAdaptStruct user;

		for(int i=0; i<m_userList.size(); i++){
			user = (_DPAdaptStruct) m_userList.get(i);
			As = user.getThetaStar().getModel();
			m_pWeights = new double[m_gWeights.length];
			for(int n=0; n<=m_featureSize; n++){
				ki = m_featureGroupMap[n];
				m_pWeights[n] = As[ki]*m_gWeights[n] + As[ki+m_dim];
			}
			user.setPersonalizedModel(m_pWeights);
		}
	}
	
	
	public void setsdB(double sd){
		m_abNuB[1] = sd;
	}
	// Assign the optimized models to the clusters.
	@Override
	protected void setThetaStars(){
		// Assign models to clusters.
		for(int i=0; i<m_kBar; i++)
			System.arraycopy(m_models, m_dim*2*i, m_thetaStars[i].getModel(), 0, m_dim*2);
	}
	
	@Override
	public String toString() {
		return String.format("CLinAdaptWithDP[dim:%d,M:%d,alpha:%.4f,#Iter:%d,N1(%.3f,%.3f),N2(%.3f,%.3f)]", m_dim,m_M, m_alpha, m_numberOfIterations, m_abNuA[0], m_abNuA[1], m_abNuB[0], m_abNuB[1]);
	}
	
	// Save the models of clusters.
	public void saveClusterModel(String modelLocation) {
		int ki;
		double[] As, pWeights;
		_thetaStar star;
		for(int k=0; k< m_kBar; k++) {
			star = m_thetaStars[k];
			try {
				File file = new File(modelLocation);
				if(!file.exists())
					file.mkdirs();
	            BufferedWriter writer = new BufferedWriter(new FileWriter(modelLocation+"/"+k+".classifier"));
	            StringBuilder buffer = new StringBuilder(512);
	            pWeights = new double[m_gWeights.length];
	            As = star.getModel();
				for(int n=0; n<=m_featureSize; n++){
					ki = m_featureGroupMap[n];
					pWeights[n] = As[ki]*m_gWeights[n] + As[ki+m_dim];
				}	            
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
		System.out.format("[Info]Save cluster models to %s.\n", modelLocation);
	}
	
	public void debug(_AdaptStruct u, int count){
		_DPAdaptStruct user = (_DPAdaptStruct) u;
		int index = findThetaStar(user.getThetaStar());
		System.out.print(String.format("\nError number:%d, ttl rvw size:%d, theta index:%d\n", count, user.getUser().getReviewSize(), index));
	}
	
//	@Override
//	public double test(){
//		m_perf = new double[m_classNo * 2];
//		m_microStat = new _PerformanceStat(m_classNo);
//
//		_AdaptStruct user;
//		_PerformanceStat userPerfStat;
//		try {
//			for (int i = 0; i <m_userList.size(); i ++) {
//				user = m_userList.get(i);
//				if ((m_testmode==TestMode.TM_batch && user.getTestSize()<1) // no testing data
//					|| (m_testmode==TestMode.TM_online && user.getAdaptationSize()<1) // no adaptation data
//					|| (m_testmode==TestMode.TM_hybrid && user.getAdaptationSize()<1) && user.getTestSize()<1) // no testing and adaptation data 
//					continue;
//					userPerfStat = user.getPerfStat();								
//				if (m_testmode==TestMode.TM_batch || m_testmode==TestMode.TM_hybrid) {				
//					//record prediction results
//					int count = 0;
//					for(_Review r:user.getReviews()) {
//						if (r.getType() != rType.TEST)
//							continue;
//						int trueL = r.getYLabel();
//						int predL = user.predict(r); // evoke user's own model
////						if(trueL == 1 && predL == 0){
////							System.out.print("fn"+"\t");
////							count++;
////						}
////						if(trueL == 0 && predL == 1){
////							System.out.print("fp"+"\t");
////							count++;
////						}
//						userPerfStat.addOnePredResult(predL, trueL);
//					}
////					if(count > 0){
////						debug(user, count);
////						System.out.println("---------------\n");
////					}
//				}							
//				userPerfStat.calculatePRF();	
//			}
//		} catch(Exception ex) {
//			ex.printStackTrace(); 
//		}		
//
//		int count = 0;
//		double[] macroF1 = new double[m_classNo];
//		
//		for(_AdaptStruct u:m_userList) {
//			if ( (m_testmode==TestMode.TM_batch && u.getTestSize()<1) // no testing data
//				|| (m_testmode==TestMode.TM_online && u.getAdaptationSize()<1) // no adaptation data
//				|| (m_testmode==TestMode.TM_hybrid && u.getAdaptationSize()<1) && u.getTestSize()<1) // no testing and adaptation data 
//				continue;
//			
//			userPerfStat = u.getPerfStat();
//			for(int i=0; i<m_classNo; i++)
//				macroF1[i] += userPerfStat.getF1(i);
//			m_microStat.accumulateConfusionMat(userPerfStat);
//			count ++;
//		}
//		
//		System.out.println(toString());
//		calcMicroPerfStat();
//		
//		// macro average
//		System.out.println("\nMacro F1:");
//		for(int i=0; i<m_classNo; i++){
//			System.out.format("Class %d\t%.4f\t", i, macroF1[i]/count);
//			m_perf[i+m_classNo] = macroF1[i]/count;
//		}
//		System.out.println();
//		return Utils.sumOfArray(macroF1);
//	}
}