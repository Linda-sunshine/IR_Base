package Classifier.supervised.modelAdaptation.CoLinAdapt;

import java.util.HashMap;
import structures._SparseFeature;
import Classifier.supervised.modelAdaptation._AdaptStruct;

/***
 * In this class, we would like to incorporate the cluster information 
 * to see how it influences the final performance. 
 * @author lin
 */
public class ClusteredLinAdapt extends LinAdapt{
	double[] m_AcAs; // The vector contains the transformation matrix for clusters and global part.
	int m_clusterNo; // The cluster number.
	// Parameters for different parts.
	double m_u = 1; // global parts.
	double m_c = 1; // cluster parts.
	double m_i = 1; // individual parts.
	int[] m_userClusterIndex; // The index is user index, the value is corresponding cluster no.

	public ClusteredLinAdapt(int classNo, int featureSize,
			HashMap<String, Integer> featureMap, String globalModel,
			String featureGroupMap, int clusterNo, int[] userClusterIndex) {
		super(classNo, featureSize, featureMap, globalModel, featureGroupMap);
		m_clusterNo = clusterNo;
		m_userClusterIndex = userClusterIndex;
	}
	
	public void setParameters(double u, double c, double i){
		m_u = u;
		m_c = c;
		m_i = i;
	}
	
	protected double linearFunc(_SparseFeature[] fvs, _AdaptStruct u) {
		
		_LinAdaptStruct user = (_LinAdaptStruct)u;
		int clusterIndex = m_userClusterIndex[user.getId()];
		double scaling, shifting;
		scaling = m_u*getGlobalScaling(0) + m_c*getClusterScaling(clusterIndex, 0) + m_i*user.getScaling(0);
		shifting = m_u*getGlobalShifting(0) + m_c*getClusterShifting(clusterIndex, 0) + m_i*user.getShifting(0);
		double value = scaling*m_gWeights[0] + shifting;//Bias term: w0*a0+b0.
		int n = 0, k = 0; // feature index and feature group index
		for(_SparseFeature fv: fvs){
			n = fv.getIndex() + 1;
			k = m_featureGroupMap[n];
			scaling = m_u*getGlobalScaling(k) + m_c*getClusterScaling(clusterIndex, k) + m_i*user.getScaling(k);
			shifting = m_u*getGlobalShifting(k) + m_c*getClusterShifting(clusterIndex, k) + m_i*user.getShifting(k);
			value += (scaling*m_gWeights[n] + shifting) * fv.getValue();
		}
		return value;
	}	
	
	public double getClusterScaling(int cNo, int k){
		return m_AcAs[k + m_dim*2*cNo];
	}
	public double getClusterShifting(int cNo, int k){
		return m_AcAs[k + m_dim + m_dim*2*cNo];
	}
	
	public double getGlobalScaling(int k){
		return m_AcAs[k + m_dim*2*m_clusterNo];
	}
	public double getGlobalShifting(int k){
		return m_AcAs[k + m_dim*2*m_clusterNo + m_dim];
	}
}
