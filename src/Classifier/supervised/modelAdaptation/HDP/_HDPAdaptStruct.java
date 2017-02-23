package Classifier.supervised.modelAdaptation.HDP;

import java.util.Collection;
import java.util.HashMap;

import Classifier.supervised.modelAdaptation.DirichletProcess.CLRWithDP;
import Classifier.supervised.modelAdaptation.DirichletProcess._DPAdaptStruct;
import structures._Doc;
import structures._HDPThetaStar;
import structures._MMBNeighbor;
import structures._Review;
import structures._SparseFeature;
import structures._User;
import utils.Utils;

public class _HDPAdaptStruct extends _DPAdaptStruct {
	
	_HDPThetaStar  m_hdpThetaStar = null;
	
	// key: global component parameter; val: member size
	protected HashMap<_HDPThetaStar, Integer> m_hdpThetaMemSizeMap;
	// key: global component parameter; val: edge size.
	protected HashMap<_HDPThetaStar, Integer> m_hdpThetaEdgeSizeMap;
	// key: uj; val: group parameter-_HDPThetaStar.
	protected HashMap<_HDPAdaptStruct, _MMBNeighbor> m_neighborMap;
	
	public _HDPAdaptStruct(_User user) {
		super(user);
		m_hdpThetaMemSizeMap = new HashMap<_HDPThetaStar, Integer>();
		m_hdpThetaEdgeSizeMap = new HashMap<_HDPThetaStar, Integer>();
		m_neighborMap = new HashMap<_HDPAdaptStruct, _MMBNeighbor>();
	}

	public _HDPAdaptStruct(_User user, int dim){
		super(user, dim);
		m_hdpThetaMemSizeMap = new HashMap<_HDPThetaStar, Integer>();
		m_hdpThetaEdgeSizeMap = new HashMap<_HDPThetaStar, Integer>();
		m_neighborMap = new HashMap<_HDPAdaptStruct, _MMBNeighbor>();
	}

	//Return the number of members in the given thetaStar.
	public double getHDPThetaMemSize(_HDPThetaStar s){
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
	
	/********Functions used in MMB model.********/
	// Return the number of edges in the given thetaStar.
	public int getHDPThetaEdgeSize(_HDPThetaStar s){
		if(m_hdpThetaEdgeSizeMap.containsKey(s))
			return m_hdpThetaEdgeSizeMap.get(s);
		else 
			return 0;
	}
	// Update the size of the edges belong to the group.
	public void incHDPThetaStarEdgeSize(_HDPThetaStar s, int v){
		if (v==0)
			return;
		
		if(m_hdpThetaEdgeSizeMap.containsKey(s))
			v += m_hdpThetaEdgeSizeMap.get(s);
		
		if (v>0)
			m_hdpThetaEdgeSizeMap.put(s, v);
		else
			m_hdpThetaEdgeSizeMap.remove(s);
	}
	
	// Check if the user has connection with another user, uj.
	public boolean hasEdge(_HDPAdaptStruct uj){
		if(m_neighborMap.containsKey(uj))
			return true;
		else
			return false;
	}
	public int getEdge(_HDPAdaptStruct uj){
		return m_neighborMap.get(uj).getEdge();
	}
	// Add a neighbor, update the <Neighbor, ThetaStar> map and <Neighbor, edge_value> map.
	public void addNeighbor(_HDPAdaptStruct uj, _HDPThetaStar theta, int e){
		m_neighborMap.put(uj, new _MMBNeighbor(uj, theta, e));
		
		// Increase the edge size by 1.
		incHDPThetaStarEdgeSize(theta, 1);
	}
	
	// Remove one neighbor, 
	public void rmNeighbor(_HDPAdaptStruct uj){
		// Decrease the edge size by 1.
		_HDPThetaStar theta = m_neighborMap.get(uj).getHDPThetaStar();
		incHDPThetaStarEdgeSize(theta, -1);
		
		m_neighborMap.remove(uj);
	}
	
	// Get the group membership for the edge between i->j.
	public _HDPThetaStar getThetaStar(_HDPAdaptStruct uj){
		return m_neighborMap.get(uj).getHDPThetaStar();
	}
	
	public _MMBNeighbor getOneNeighbor(_HDPAdaptStruct u){
		return m_neighborMap.get(u);
	}
	public HashMap<_HDPAdaptStruct, _MMBNeighbor> getNeighbors(){
		return m_neighborMap;
	}
	
	/**********************/

	public Collection<_HDPThetaStar> getHDPTheta(){
		return m_hdpThetaMemSizeMap.keySet();
	}
	
	public void setThetaStar(_HDPThetaStar theta){
		m_hdpThetaStar = theta;
	}
	@Override
	public _HDPThetaStar getThetaStar(){
		return m_hdpThetaStar;
	}
	
	@Override
	public double evaluate(_Doc doc) {
		_Review r = (_Review) doc;
		double prob = 0, sum = 0;
		double[] probs = r.getCluPosterior();
		int n, m, k;

		//not adaptation based
		if (m_dim==0) {
			for(k=0; k<probs.length; k++) {
				sum = Utils.dotProduct(CLRWithHDP.m_hdpThetaStars[k].getModel(), doc.getSparse(), 0);//need to be fixed: here we assumed binary classification
				if(MTCLRWithHDP.m_supWeights != null && MTCLRWithHDP.m_q != 0)
					sum += CLRWithDP.m_q*Utils.dotProduct(MTCLRWithHDP.m_supWeights, doc.getSparse(), 0);
								
				//to maintain numerical precision, compute the expectation in log space as well
				if (k==0)
					prob = probs[k] + Math.log(Utils.logistic(sum));
				else
					prob = Utils.logSum(prob, probs[k] + Math.log(Utils.logistic(sum)));
			}
		} else {
			double As[];
			for(k=0; k<probs.length; k++) {
				As = CLRWithHDP.m_hdpThetaStars[k].getModel();
				sum = As[0]*CLinAdaptWithHDP.m_supWeights[0] + As[m_dim];//Bias term: w_s0*a0+b0.
				for(_SparseFeature fv: doc.getSparse()){
					n = fv.getIndex() + 1;
					m = m_featureGroupMap[n];
					sum += (As[m]*CLinAdaptWithHDP.m_supWeights[n] + As[m_dim+m]) * fv.getValue();
				}
				
				//to maintain numerical precision, compute the expectation in log space as well
				if (k==0)
					prob = probs[k] + Math.log(Utils.logistic(sum));
				else
					prob = Utils.logSum(prob, probs[k] + Math.log(Utils.logistic(sum)));
			}
		}
		
		//accumulate the prediction results during sampling procedure
		doc.m_pCount ++;
		doc.m_prob += Math.exp(prob); //>0.5?1:0;
		return prob;
	}	
	
	public double evaluateTrain(_Doc doc){
		_Review r = (_Review) doc;
		double prob = 0, sum = 0;
		double[] probs = r.getCluPosterior();
		int n, m, k;

		//not adaptation based
		if (m_dim==0) {
			for(k=0; k<probs.length; k++) {
//				sum = Utils.dotProduct(CLRWithHDP.m_hdpThetaStars[k].getModel(), doc.getSparse(), 0);//need to be fixed: here we assumed binary classification
				if(MTCLRWithHDP.m_supWeights != null && MTCLRWithHDP.m_q != 0)
					sum += CLRWithDP.m_q*Utils.dotProduct(MTCLRWithHDP.m_supWeights, doc.getSparse());
				
				//to maintain numerical precision, compute the expectation in log space as well
				if (k==0)
					prob = probs[k] + Math.log(Utils.logistic(sum));
				else
					prob = Utils.logSum(prob, probs[k] + Math.log(Utils.logistic(sum)));
			}
		} else {
			double As[];
			for(k=0; k<probs.length; k++) {
				As = CLRWithHDP.m_hdpThetaStars[k].getModel();
				sum = As[0]*CLinAdaptWithHDP.m_supWeights[0] + As[m_dim];//Bias term: w_s0*a0+b0.
				for(_SparseFeature fv: doc.getSparse()){
					n = fv.getIndex() + 1;
					m = m_featureGroupMap[n];
					sum += (As[m]*CLinAdaptWithHDP.m_supWeights[n] + As[m_dim+m]) * fv.getValue();
				}
				
				//to maintain numerical precision, compute the expectation in log space as well
				if (k==0)
					prob = probs[k] + Math.log(Utils.logistic(sum));
				else
					prob = Utils.logSum(prob, probs[k] + Math.log(Utils.logistic(sum)));
			}
		}
		
		//accumulate the prediction results during sampling procedure
		doc.m_pTrainCount ++;
		doc.m_probTrain += Math.exp(prob); //>0.5?1:0;
		return prob;
	}
}
