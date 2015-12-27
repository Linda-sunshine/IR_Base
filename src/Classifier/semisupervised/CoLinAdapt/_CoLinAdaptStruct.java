/**
 * 
 */
package Classifier.semisupervised.CoLinAdapt;

import java.util.Vector;

import structures.MyPriorityQueue;
import structures._RankItem;
import structures._User;

/**
 * @author Hongning Wang
 * Add the shared structure for efficiency purpose
 */
public class _CoLinAdaptStruct extends _LinAdaptStruct {
	public enum SimType {
		ST_BoW,
		ST_SVD
	}
	
	static double[] sharedA;
	int m_id;
	MyPriorityQueue<_RankItem> m_neighbors;//top-K neighborhood, this graph structure is asymmetric
	
	public _CoLinAdaptStruct(_User user, int dim, int id, int topK) {
		super(user, dim);
		m_id = id;
		m_neighbors = new MyPriorityQueue<_RankItem>(topK);
	}
	
	public void addNeighbor(int id, double similarity) {
		m_neighbors.add(new _RankItem(id, similarity));
	}
	
	public double getSimilarity(_CoLinAdaptStruct user, SimType sType) {
		if (sType == SimType.ST_BoW)
			return user.m_user.getBoWSim(m_user);
		else
			return user.m_user.getSVDSim(m_user);
	}
	
	public Vector<_RankItem> getNeighborhood() {
		return m_neighbors;
	}

	static public double[] getSharedA() {
		return sharedA;
	}
	
	//this operation becomes very expensive in _CoLinStruct
	@Override
	public double[] getA() {
		int offset = m_id * m_dim * 2;
		System.arraycopy(sharedA, offset, m_A, 0, m_dim*2);
		return m_A;
	}
	
	//get the shifting operation for this group
	@Override
	public double getShifting(int gid) {
		int offset = m_id * m_dim * 2;
		return sharedA[offset+m_dim+gid];
	}
	
	//get the shifting operation for this group
	@Override
	public double getScaling(int gid) {
		int offset = m_id * m_dim * 2;
		return sharedA[offset+gid];
	}
}
