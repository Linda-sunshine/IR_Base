package Classifier.supervised.modelAdaptation.HDP;

import java.util.Collection;
import java.util.HashMap;

import Classifier.supervised.modelAdaptation.DirichletProcess._DPAdaptStruct;
import structures._Doc;
import structures._HDPThetaStar;
import structures._Review;
import structures._User;
import utils.Utils;

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

	//will remove the key if the updated value is zero
	public void incHDPThetaStarMemSize(_HDPThetaStar s, int v){
		if (v==0)
			return;
		
		if(m_hdpThetaMemSizeMap.containsKey(s))
			v += m_hdpThetaMemSizeMap.get(s);
		
		if (v>0)
			m_hdpThetaMemSizeMap.put(s, v);
		else
			m_hdpThetaMemSizeMap.remove(s);
	}
	
	public Collection<_HDPThetaStar> getHDPTheta(){
		return m_hdpThetaMemSizeMap.keySet();
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
				if(m_supModel != null)
					sum += Utils.dotProduct(m_supModel, doc.getSparse())*m_q;
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
