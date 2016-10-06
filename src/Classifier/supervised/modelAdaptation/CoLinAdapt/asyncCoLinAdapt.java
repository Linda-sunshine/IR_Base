/**
 * 
 */
package Classifier.supervised.modelAdaptation.CoLinAdapt;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedList;

import Classifier.supervised.modelAdaptation._AdaptStruct;
import Classifier.supervised.modelAdaptation._AdaptStruct.SimType;
import Classifier.supervised.modelAdaptation.RegLR.asyncRegLR;
import structures._PerformanceStat;
import structures._PerformanceStat.TestMode;
import structures._Review.rType;
import structures._RankItem;
import structures._Review;
import structures._UserReviewPair;

/**
 * @author Hongning Wang
 * asynchronized CoLinAdapt with zero order gradient update, i.e., we will only touch the current user's gradient
 */
public class asyncCoLinAdapt extends CoLinAdapt {
	double m_initStepSize = 0.01;
	boolean m_trainByUser = false; // by default we will perform online training by review timestamp; otherwise we will do it by user. 

	int[] m_userOrder; // visiting order of different users during online learning
	PrintWriter m_writer;
	int m_rptTime = 3, m_count = 0; // How many times the reviews will be used to update gradients.

	public asyncCoLinAdapt(int classNo, int featureSize, HashMap<String, Integer> featureMap, int topK, String globalModel, String featureGroupMap) {
		super(classNo, featureSize, featureMap, topK, globalModel, featureGroupMap);
		
		// all three test modes for asyncCoLinAdapt is possible, and default is online
		m_testmode = TestMode.TM_online;
	}
	
	@Override
	public String toString() {
		return String.format("asyncCoLinAdapt[dim:%d,eta1:%.3f,eta2:%.3f,eta3:%.3f,eta4:%.3f,k:%d,NB:%s]", m_dim, m_eta1, m_eta2, m_eta3, m_eta4, m_topK, m_sType);
	}
	
	public void setInitStepSize(double initStepSize) {
		m_initStepSize = initStepSize;
	}
	// Set the repeat time for the training.
	public void setRPTTime(int t){
		m_rptTime = t;
	}
	
	public void resetRPTTime(){
		m_count = m_rptTime;
	}
	public void setTrainByUser(boolean b){
		m_trainByUser = b;
	}
	
	@Override
	protected void constructNeighborhood(SimType sType) {
		super.constructNeighborhood(sType);
		m_userOrder = constructReverseNeighborhood();
	}
	
	@Override
	protected void gradientByFunc(_AdaptStruct user) {		
		//Update gradients review by review within the latest adaptation cache.
		for(_Review review:user.nextAdaptationIns())
			gradientByFunc(user, review, 1.0); // equal weights for this user's own adaptation data
	}
	
	@Override
	protected void gradientByR2(_AdaptStruct user){		
		_CoLinAdaptStruct uj, ui = (_CoLinAdaptStruct)user;
		
		for(_RankItem nit:ui.getNeighbors()) {
			uj = (_CoLinAdaptStruct)m_userList.get(nit.m_index);
			gradientByR2(ui, uj, nit.m_value);
		}
		
		for(_RankItem nit:ui.getReverseNeighbors()) {
			uj = (_CoLinAdaptStruct)m_userList.get(nit.m_index);
			gradientByR2(ui, uj, nit.m_value);
		}
	}
	
	//we will only update ui but keep uj as constant
	void gradientByR2(_CoLinAdaptStruct ui, _CoLinAdaptStruct uj, double sim) {
		double coef = 2 * sim, dA, dB;
		int offset = m_dim*2*ui.getId();
		
		for(int k=0; k<m_dim; k++) {
			dA = coef * m_eta3 * (ui.getScaling(k) - uj.getScaling(k));
			dB = coef * m_eta4 * (ui.getShifting(k) - uj.getShifting(k));
			
			// update ui's gradient
			m_g[offset + k] += dA;
			m_g[offset + k + m_dim] += dB;
		}
	}
	
	protected double gradientTest(_AdaptStruct user) {
		int offset, uid = 2*m_dim*user.getId();
		double magA = 0, magB = 0;
		
		for(int i=0; i<m_dim; i++){
			offset = uid + i;
			magA += m_g[offset]*m_g[offset];
			magB += m_g[offset+m_dim]*m_g[offset+m_dim];
		}
		if(Double.isNaN(magA) || Double.isNaN(magB))
			System.out.println("NaN detected here!!");
		
		if (m_displayLv==2)
			System.out.format("Gradient magnitude for a: %.5f, b: %.5f\n", magA, magB);
		return magA + magB;
	}
	
	@Override
	protected int getAdaptationSize(_AdaptStruct user) {
		return user.getAdaptationCacheSize();
	}

	//this is online training in each individual user
	@Override
	public double train(){
		initLBFGS();
		init();
		
		if (m_trainByUser)
			trainByUser();
		else
			trainByReview();
		
		setPersonalizedModel();
		return 0;//we do not evaluate function value
	}
	
	
	public void trainByReview(){
		LinkedList<_UserReviewPair> reviewlist = new LinkedList<_UserReviewPair>();
		
		double gNorm, gNormOld = Double.MAX_VALUE;
		int predL, trueL, counter = 0;
		_Review doc;
		double val = 0;
		_CoLinAdaptStruct user;
		_PerformanceStat perfStat;
		
		try{
		m_writer = new PrintWriter(new File(String.format("CoLinAdapt_OnlineByReview_%d.txt", m_rptTime)));
		//collect the training/adaptation data
		for(int i=0; i<m_userList.size(); i++) {
			user = (_CoLinAdaptStruct)m_userList.get(i);
			for(_Review r:user.getReviews()) {
				if (r.getType() == rType.ADAPTATION || r.getType() == rType.TRAIN)
					reviewlist.add(new _UserReviewPair(user, r));//we will only collect the training or adaptation reviews
			}
		}
		
		//sort them by timestamp
		Collections.sort(reviewlist);
		
		for(_UserReviewPair pair:reviewlist) {
			user = (_CoLinAdaptStruct)pair.getUser();
			// test the latest model before model adaptation
			if (m_testmode != TestMode.TM_batch) {
				doc = pair.getReview();
				perfStat = user.getPerfStat();
				val = logit(doc.getSparse(), user);
				predL = predict(doc, user);
				trueL = doc.getYLabel();
				perfStat.addOnePredResult(predL, trueL);
				m_writer.format("%s\t%d\t%.4f\t%d\t%d\n", user.getUserID(), doc.getID(), val, predL, trueL);
			}// in batch mode we will not accumulate the performance during adaptation	
			
			// prepare to adapt: initialize gradient	
			Arrays.fill(m_g, 0); 
			calculateGradients(user);
			gNorm = gradientTest(user);
			
			if (m_displayLv==1) {
				if (gNorm<gNormOld)
					System.out.print("o");
				else
					System.out.print("x");
			}
			
			//gradient descent
			gradientDescent(user, m_initStepSize, 1.0);
			//gradientDescent(user, asyncLinAdapt.getStepSize(initStepSize, user));
			gNormOld = gNorm;
			
			if (m_displayLv>0 && ++counter%100==0)
				System.out.println();
		}
		m_writer.close();
		} catch(IOException e){
			e.printStackTrace();
		}
	}
	
	//this is online training in each individual user
	public void trainByUser(){
		double gNorm, gNormOld = Double.MAX_VALUE;
		int counter = 0;
		_CoLinAdaptStruct user;
		int predL, trueL;
		_Review doc;
		_PerformanceStat perfStat;
		double val = 0;
		initLBFGS();
		init();
		try{
		m_writer = new PrintWriter(new File(String.format("CoLinAdapt_OnlineByUser_%d.txt", m_rptTime)));

		for(int t=0; t<m_userOrder.length; t++) {
			user = (_CoLinAdaptStruct)m_userList.get(m_userOrder[t]);

			if(user.hasNextAdaptationIns()) {
				// test the latest model
				if (m_testmode!=TestMode.TM_batch && (doc = user.getLatestTestIns()) != null) {
					perfStat = user.getPerfStat();
					val = logit(doc.getSparse(), user);
					predL = predict(doc, user);
					trueL = doc.getYLabel();
					perfStat.addOnePredResult(predL, trueL);
					m_writer.format("%s\t%d\t%.4f\t%d\t%d\n", user.getUserID(), doc.getID(), val, predL, trueL);
				} // in batch mode we will not accumulate the performance during adaptation			
				
				// prepare to adapt: initialize gradient	
				Arrays.fill(m_g, 0); 
				calculateGradients(user);
				gNorm = gradientTest(user);
				
				if (m_displayLv==1) {
					if (gNorm<gNormOld)
						System.out.print("o");
					else
						System.out.print("x");
				}
				
				//gradient descent
				gradientDescent(user, m_initStepSize, 1.0);
				//gradientDescent(user, asyncLinAdapt.getStepSize(initStepSize, user));
				gNormOld = gNorm;
				
				if (m_displayLv>0 && ++counter%100==0)
					System.out.println();
			}			
		}
		m_writer.close();
		} catch(IOException e){
			e.printStackTrace();
		}
	}
		
	// update this current user only
	void gradientDescent(_CoLinAdaptStruct user, double initStepSize, double inc) {
		double a, b, stepSize = asyncRegLR.getStepSize(initStepSize, user);
		int offset = 2*m_dim*user.getId();
		
		//Added by Lin for reusing the reviews.
		resetRPTTime();
		while(m_count-- > 0){
			for(int k=0; k<m_dim; k++) {
				a = user.getScaling(k) - stepSize * m_g[offset + k];
				user.setScaling(k, a);
			
				b = user.getShifting(k) - stepSize * m_g[offset + k + m_dim];
				user.setShifting(k, b);
			}
//			user.incUpdatedCount(inc);
			
			//update the record of updating history
			if(m_count == 0)
				user.incUpdatedCount(inc);
		}
	}	
}
