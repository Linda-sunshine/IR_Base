package Classifier.supervised.modelAdaptation.CoLinAdapt;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedList;

import structures._PerformanceStat;
import structures._Review;
import structures._PerformanceStat.TestMode;
import structures._Review.rType;
import structures._User;
import structures._UserReviewPair;
import utils.Utils;
import Classifier.supervised.modelAdaptation._AdaptStruct;
import Classifier.supervised.modelAdaptation.RegLR.asyncRegLR;

public class asyncMTLinAdapt extends MTLinAdapt{

	double m_initStepSize = 0.25;
	boolean m_trainByUser = true; 
	PrintWriter m_writer;

	String m_dataset;
	
	public asyncMTLinAdapt(int classNo, int featureSize,
			HashMap<String, Integer> featureMap, int topK, String globalModel,
			String featureGroupMap, String featureGroupMap4Sup) {
		super(classNo, featureSize, featureMap, topK, globalModel, featureGroupMap);
		loadFeatureGroupMap4SupUsr(featureGroupMap4Sup);
		m_testmode = TestMode.TM_online;
	}
	
	@Override
	public String toString() {
		return String.format("asyncMTLinAdapt[dim:%d,SupDim:%d, eta1:%.3f,eta2:%.3f, lambda1:%.3f, lambda2:%.3f]", m_dim, m_dimSup, m_eta1, m_eta2, m_lambda1, m_lambda2);
	}
	public void setTrainByUser(boolean b){
		m_trainByUser = b;
	}
	
	public void setDataset(String data){
		m_dataset = data;
	}
	
	@Override
	protected void init(){
		super.init();
	 		
	 	// this is also incorrect, since we should normalized it by total number of adaptation reviews
	 	m_lambda1 /= m_userSize;
	 	m_lambda2 /= m_userSize;
	}
	protected void calculateGradients(_AdaptStruct u){
		gradientByFunc(u);
		gradientByR1(u);
		gradientByRs();
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
	
	void trainByReview() {
		LinkedList<_UserReviewPair> reviewlist = new LinkedList<_UserReviewPair>();
		
		double gNorm, gNormOld = Double.MAX_VALUE;
		int predL, trueL, counter = 0;
		double val = 0;
		_Review doc;
		_CoLinAdaptStruct user;
		int count = 0;
		try{
			m_writer = new PrintWriter(new File(String.format("%s_online_trainByReview.txt", m_dataset)));
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
					val = logit(doc.getSparse(), user);
					predL = predict(doc, user);
					trueL = doc.getYLabel();
					user.getPerfStat().addOnePredResult(predL, trueL);
					m_writer.format("%s\t%d\t%.4f\t%d\t%d\t", user.getUserID(), doc.getID(), val, predL, trueL);
				}// in batch mode we will not accumulate the performance during adaptation	
					
				gradientDescent(user, m_initStepSize, 1.0);
				
				m_writer.format("%.6f\t%.6f\n", calcDiffWsWg(), calcDiffWsWi(user));
				//test the gradient only when we want to debug
				if (m_displayLv>0) {
					gNorm = gradientTest();				
					if (m_displayLv==1) {
						if (gNorm<gNormOld)
							System.out.print("o");
						else
							System.out.print("x");
					}				
					gNormOld = gNorm;
					if (++counter%120==0)
						System.out.println();
				}
				count++;
			}
			m_writer.close();
		} catch(IOException e){
				e.printStackTrace();
		}
	}
	
	void trainByUser() {
		double gNorm, gNormOld = Double.MAX_VALUE;
		int predL, trueL;
		double val = 0;
		_Review doc;
		_CoLinAdaptStruct user;
		try{
			m_writer = new PrintWriter(new File("online_trainByUser.txt"));
			for(int i=0; i<m_userList.size(); i++) {
				user = (_CoLinAdaptStruct)m_userList.get(i);
			
				while(user.hasNextAdaptationIns()) {
					// test the latest model before model adaptation
					if (m_testmode != TestMode.TM_batch && (doc = user.getLatestTestIns()) != null) {
						
						val = logit(doc.getSparse(), user);
						predL = val>0.5?1:0;
						trueL = doc.getYLabel();
						user.getPerfStat().addOnePredResult(predL, trueL);
						m_writer.format("%s\t%d\t%.4f\t%d\t%d\n", user.getUserID(), doc.getID(), val, predL, trueL);
					} // in batch mode we will not accumulate the performance during adaptation				
				
					gradientDescent(user, m_initStepSize, 1.0);
				
					//test the gradient only when we want to debug
					if (m_displayLv>0) {
						gNorm = gradientTest();				
						if (m_displayLv==1) {
							if (gNorm<gNormOld)
								System.out.print("o");
							else
								System.out.print("x");
						}				
						gNormOld = gNorm;
					}
				}
			} 
			if (m_displayLv==1)
				System.out.println();
			m_writer.close();
		} catch(IOException e){
			e.printStackTrace();
		}
	}
	
	@Override
	protected int getAdaptationSize(_AdaptStruct user) {
		return user.getAdaptationCacheSize();
	}
	
	@Override
	protected void gradientByFunc(_AdaptStruct user) {		
		//Update gradients one review by one review.
		for(_Review review:user.nextAdaptationIns())
			gradientByFunc(user, review, 1.0);//equal weight for the user's own adaptation data
	}
	
	// update this current user only
	void gradientDescent(_CoLinAdaptStruct user, double initStepSize, double inc) {
		double a, b, stepSize = asyncRegLR.getStepSize(initStepSize, user);
		int offset = 2 * m_dim * user.getId();
		int supOffset = 2 * m_dim * m_userList.size();
		
		//get gradient
		Arrays.fill(m_g, 0);
		calculateGradients(user);
		
		for (int k = 0; k < m_dim; k++) {
			a = user.getScaling(k) - stepSize * m_g[offset + k];
			user.setScaling(k, a);

			b = user.getShifting(k) - stepSize * m_g[offset + k + m_dim];
			user.setShifting(k, b);
		}
		
		stepSize /= 3;
		//update the super user
		for(int k=0; k<m_dimSup; k++) {
			m_A[supOffset+k] -= stepSize * m_g[supOffset + k];
			m_A[supOffset+k+m_dimSup] -= stepSize * m_g[supOffset + k + m_dimSup];
		}
		
		//update the record of updating history
		user.incUpdatedCount(inc);
	}
	
	public void loadGlobal(String filename){
		if (filename==null)
			return;	
		try{
			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(filename), "UTF-8"));
			String line;
			int pos = 0;
			m_gWeights = new double[m_featureSize+1];//to include the bias term
			while((line=reader.readLine()) != null) {
				m_gWeights[pos++] = Double.valueOf(line);
			}
			reader.close();
		} catch(IOException e){
			System.err.format("[Error]Fail to open file %s.\n", filename);
		}
	}
	
	public double calcDiffWsWg(){
		int gid;
		// Get a copy of super user's transformation matrix.
		double[] As = Arrays.copyOfRange(m_A, m_userList.size()*m_dim*2, m_userList.size()*m_dim*2 + m_dimSup*2);
		
		// Set the bias term for ws.
		m_sWeights[0] = As[0] * m_gWeights[0] + As[m_dimSup];
		// Set the other terms for ws.
		for(int n=0; n<m_featureSize; n++){
			gid = m_featureGroupMap4SupUsr[1+n];
			m_sWeights[n+1] = As[gid] * m_gWeights[1+n] + As[gid+ m_dimSup];
		}
		return Utils.EuclideanDistance(m_gWeights, m_sWeights);
	}
	
	public double calcDiffWsWi(_AdaptStruct user){
		_CoLinAdaptStruct ui = (_CoLinAdaptStruct) user;
		int gid;
					
		//set bias term
		m_pWeights[0] = ui.getScaling(0) * m_sWeights[0] + ui.getShifting(0);
		//set the other features
		for(int n=0; n<m_featureSize; n++) {
			gid = m_featureGroupMap[1+n];
			m_pWeights[1+n] = ui.getScaling(gid) * m_sWeights[1+n] + ui.getShifting(gid);
		}
		return Utils.EuclideanDistance(m_pWeights, m_sWeights);
	}
}
