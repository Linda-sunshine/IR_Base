package Classifier.supervised.modelAdaptation.HDP;

import java.util.HashMap;

import structures._Doc;
import structures._HDPThetaStar;
import structures._Review;
import structures._User;
import utils.Utils;
import Classifier.supervised.modelAdaptation.DirichletProcess._DPAdaptStruct;

public class _HDPAdaptStruct extends _DPAdaptStruct {
	
	_HDPThetaStar  m_hdpThetaStar = null;
	
	/**key: global component parameter \phi; 
	 * val: member size.**/
	protected HashMap<_HDPThetaStar, Integer> m_hdpThetaMemSizeMap;
	
	public _HDPAdaptStruct(_User user) {
		super(user);
		m_hdpThetaMemSizeMap = new HashMap<_HDPThetaStar, Integer>();
	}

	//Return the number of members in the given thetaStar.
	public int getHDPThetaMemSize(_HDPThetaStar s){
		if(m_hdpThetaMemSizeMap.containsKey(s))
			return m_hdpThetaMemSizeMap.get(s);
		else 
			return 0;
	}

	public void updateHDPThetaStarMemSize(_HDPThetaStar s, int v){
		if(!m_hdpThetaMemSizeMap.containsKey(s))
			m_hdpThetaMemSizeMap.put(s, 0);
		int val = m_hdpThetaMemSizeMap.get(s)+v;
		m_hdpThetaMemSizeMap.put(s, val);
	}
	
	public void rmThetaFromMemSizeMap(_HDPThetaStar s){
		if(!m_hdpThetaMemSizeMap.containsKey(s))
			System.out.println("Does not exist in size map!");
		else 
			m_hdpThetaMemSizeMap.remove(s);
	}
	
//	public void rmThetaFromHMap(_HDPThetaStar s){
//		if(!m_hMap.containsKey(s))
//			System.out.println("Does not exist in h map!");
//		else
//			m_hMap.remove(s);
//	}
	
	public HashMap<_HDPThetaStar, Integer> getHDPThetaMemSizeMap(){
		return m_hdpThetaMemSizeMap;
	}
	
	@Override
	public _HDPThetaStar getThetaStar(){
		return m_hdpThetaStar;
	}
	
	@Override
	public double evaluate(_Doc doc) {
		_Review r = (_Review) doc;
		double prob = 0, sum;
		double[] probs = r.getCluPosterior();
		if (m_dim==0) {//not adaptation based
			for(int k=0; k<probs.length; k++) {
				sum = Utils.dotProduct(CLRWithHDP.m_hdpThetaStars[k].getModel(), doc.getSparse(), 0);//need to be fixed: here we assumed binary classification
				prob += probs[k] * Utils.logistic(sum); 
			}			
//		} else {
//			int n, m;
//			double As[];
//			for(int k=0; k<m_cluPosterior.length; k++) {
//				As = CLRWithHDP.m_hdpThetaStars[k].getModel();
//
//				sum = As[0]*CLinAdaptWithHDP.m_supWeights[0] + As[m_dim];//Bias term: w_s0*a0+b0.
//				for(_SparseFeature fv: doc.getSparse()){
//					n = fv.getIndex() + 1;
//					m = m_featureGroupMap[n];
//					sum += (As[m]*CLinAdaptWithHDP.m_supWeights[n] + As[m_dim+m]) * fv.getValue();
//				}
//				
//				prob += m_cluPosterior[k] * Utils.logistic(sum); 
//			}
		}
		//accumulate the prediction results during sampling procedure
		doc.m_pCount ++;
		doc.m_prob += prob; //>0.5?1:0;
		
		return prob;
	}	
}
